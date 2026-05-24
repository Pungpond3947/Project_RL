"""Dataset utilities for one-leg actuator calibration.

The expected CSV schema is:
    t, q_0..q_2, dq_0..dq_2, cmd_0..cmd_2, q_next_0..q_next_2, dq_next_0..dq_next_2
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


_JOINTS = 3
_REQUIRED_COLUMNS = (
    ["t"]
    + [f"q_{i}" for i in range(_JOINTS)]
    + [f"dq_{i}" for i in range(_JOINTS)]
    + [f"cmd_{i}" for i in range(_JOINTS)]
    + [f"q_next_{i}" for i in range(_JOINTS)]
    + [f"dq_next_{i}" for i in range(_JOINTS)]
)


@dataclass
class UANHardwareDataset:
    """Contiguous hardware transitions loaded onto a torch device."""

    t: torch.Tensor
    q: torch.Tensor
    dq: torch.Tensor
    cmd: torch.Tensor
    tau: torch.Tensor | None
    q_next: torch.Tensor
    dq_next: torch.Tensor
    source_id: torch.Tensor
    valid_start_indices: torch.Tensor
    source_files: tuple[str, ...]
    source_lengths: tuple[int, ...]
    dt_median: float

    @classmethod
    def load(
        cls,
        csv_paths: Sequence[str | Path],
        *,
        episode_steps: int,
        device: str | torch.device = "cpu",
    ) -> "UANHardwareDataset":
        """Load one or more CSV logs and build valid rollout starts.

        Args:
            csv_paths: CSV files. Relative paths are resolved against the current
                directory first, then the repository root.
            episode_steps: Minimum contiguous rollout length needed by the RL env.
            device: Torch device for returned tensors.
        """
        if episode_steps < 1:
            raise ValueError(f"episode_steps must be positive, got {episode_steps}.")

        arrays: list[dict[str, np.ndarray]] = []
        source_files: list[str] = []
        source_lengths: list[int] = []
        offsets: list[tuple[int, int]] = []
        cursor = 0

        for raw_path in csv_paths:
            path = _resolve_csv_path(raw_path)
            data = _load_csv(path)
            length = int(data["q"].shape[0])
            if length <= episode_steps + 1:
                raise ValueError(
                    f"{path} has only {length} rows, but episode_steps={episode_steps} needs at least "
                    f"{episode_steps + 2} rows."
                )
            arrays.append(data)
            source_files.append(str(path))
            source_lengths.append(length)
            offsets.append((cursor, cursor + length))
            cursor += length

        t = np.concatenate([a["t"] for a in arrays], axis=0)
        q = np.concatenate([a["q"] for a in arrays], axis=0)
        dq = np.concatenate([a["dq"] for a in arrays], axis=0)
        cmd = np.concatenate([a["cmd"] for a in arrays], axis=0)
        tau = np.concatenate([a["tau"] for a in arrays], axis=0) if arrays[0]["tau"] is not None else None
        q_next = np.concatenate([a["q_next"] for a in arrays], axis=0)
        dq_next = np.concatenate([a["dq_next"] for a in arrays], axis=0)
        source_id = np.concatenate(
            [np.full(length, source_idx, dtype=np.int64) for source_idx, length in enumerate(source_lengths)], axis=0
        )

        valid_starts = []
        for begin, end in offsets:
            # Keep every rollout inside one file and leave one row for q_next/dq_next.
            valid_starts.append(np.arange(begin, end - episode_steps - 1, dtype=np.int64))
        valid_start_indices = np.concatenate(valid_starts, axis=0)
        if valid_start_indices.size == 0:
            raise ValueError("No valid rollout starts were found. Use a shorter episode length.")

        finite_mask = np.isfinite(q).all(axis=1) & np.isfinite(dq).all(axis=1) & np.isfinite(cmd).all(axis=1)
        if tau is not None:
            finite_mask &= np.isfinite(tau).all(axis=1)
        if not bool(finite_mask.all()):
            bad_count = int((~finite_mask).sum())
            raise ValueError(f"Hardware dataset contains {bad_count} rows with NaN/Inf values.")

        dt_values = []
        for a in arrays:
            if len(a["t"]) > 1:
                dt_values.append(np.diff(a["t"]))
        dt_median = float(np.median(np.concatenate(dt_values))) if dt_values else 0.0

        return cls(
            t=torch.as_tensor(t, dtype=torch.float64, device=device),
            q=torch.as_tensor(q, dtype=torch.float32, device=device),
            dq=torch.as_tensor(dq, dtype=torch.float32, device=device),
            cmd=torch.as_tensor(cmd, dtype=torch.float32, device=device),
            tau=torch.as_tensor(tau, dtype=torch.float32, device=device) if tau is not None else None,
            q_next=torch.as_tensor(q_next, dtype=torch.float32, device=device),
            dq_next=torch.as_tensor(dq_next, dtype=torch.float32, device=device),
            source_id=torch.as_tensor(source_id, dtype=torch.long, device=device),
            valid_start_indices=torch.as_tensor(valid_start_indices, dtype=torch.long, device=device),
            source_files=tuple(source_files),
            source_lengths=tuple(source_lengths),
            dt_median=dt_median,
        )

    @property
    def num_rows(self) -> int:
        return int(self.q.shape[0])

    @property
    def num_valid_starts(self) -> int:
        return int(self.valid_start_indices.shape[0])

    def sample_starts(self, count: int, device: str | torch.device | None = None) -> torch.Tensor:
        """Sample valid global row indices for rollout starts."""
        if device is None:
            device = self.valid_start_indices.device
        choices = torch.randint(self.num_valid_starts, (count,), device=device)
        return self.valid_start_indices[choices]


def _resolve_csv_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute() and path.exists():
        return path
    if path.exists():
        return path.resolve()

    repo_root = Path(__file__).resolve().parents[4]
    repo_path = repo_root / path
    if repo_path.exists():
        return repo_path.resolve()

    raise FileNotFoundError(f"Could not find CSV file: {raw_path}")


def _load_csv(path: Path) -> dict[str, np.ndarray]:
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64, encoding=None)
    if table.ndim == 0:
        table = table.reshape(1)

    names = table.dtype.names or ()
    missing = [name for name in _REQUIRED_COLUMNS if name not in names]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    def stack(prefix: str) -> np.ndarray:
        return np.stack([table[f"{prefix}_{i}"] for i in range(_JOINTS)], axis=-1).astype(np.float32)

    order = np.argsort(table["t"])
    has_tau = all(f"tau_{i}" in names for i in range(_JOINTS))
    return {
        "t": np.asarray(table["t"], dtype=np.float64)[order],
        "q": stack("q")[order],
        "dq": stack("dq")[order],
        "tau": stack("tau")[order] if has_tau else None,
        "cmd": stack("cmd")[order],
        "q_next": stack("q_next")[order],
        "dq_next": stack("dq_next")[order],
    }
