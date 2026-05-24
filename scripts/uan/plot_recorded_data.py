"""Plot recorded one-leg UAN CSV logs.

The script creates one PNG per CSV file. Each figure shows the recorded
position command, measured joint position, measured joint velocity, and torque
when torque columns exist in the CSV.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


JOINTS = 3
DEFAULT_OUTPUT_DIR = "logs/uan_dataset_plots"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv",
        nargs="*",
        help="CSV logs to plot. If omitted, all real_data/*.csv files are plotted.",
    )
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Directory for generated PNG files.")
    parser.add_argument("--start_s", type=float, default=0.0, help="Start time in seconds from the beginning of each log.")
    parser.add_argument("--duration_s", type=float, default=None, help="Optional duration in seconds to plot.")
    parser.add_argument(
        "--max_points",
        type=int,
        default=8000,
        help="Maximum plotted points per file after downsampling. Use 0 to disable.",
    )
    parser.add_argument(
        "--title_prefix",
        default="",
        help="Optional text prepended to each figure title.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    csv_paths = [_resolve_path(path, repo_root) for path in args.csv]
    if not csv_paths:
        csv_paths = sorted((repo_root / "real_data").glob("*.csv"))

    output_dir = _resolve_path(args.output_dir, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in csv_paths:
        data = _load_csv(csv_path)
        view = _slice_by_time(data, args.start_s, args.duration_s)
        if view["t"].size == 0:
            print(f"[WARN] No rows selected from {csv_path}")
            continue
        view = _downsample(view, args.max_points)
        output_path = output_dir / f"{csv_path.stem}.png"
        _write_plot(view, csv_path, output_path, args.title_prefix)
        duration = float(view["t"][-1] - view["t"][0]) if view["t"].size > 1 else 0.0
        print(f"[INFO] Wrote {output_path} ({view['t'].size} points, {duration:.2f} s)")


def _resolve_path(raw_path: str | Path, repo_root: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _load_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames or []
        required = ["t"] + [f"q_{i}" for i in range(JOINTS)] + [f"dq_{i}" for i in range(JOINTS)]
        required += [f"cmd_{i}" for i in range(JOINTS)]
        missing = [name for name in required if name not in fieldnames]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")

        has_tau = all(f"tau_{i}" in fieldnames for i in range(JOINTS))
        rows = list(reader)

    def column(name: str) -> np.ndarray:
        return np.asarray([float(row[name]) for row in rows], dtype=np.float64)

    data = {
        "t": column("t"),
        "q": np.stack([column(f"q_{i}") for i in range(JOINTS)], axis=-1),
        "dq": np.stack([column(f"dq_{i}") for i in range(JOINTS)], axis=-1),
        "cmd": np.stack([column(f"cmd_{i}") for i in range(JOINTS)], axis=-1),
        "tau": None,
    }
    if has_tau:
        data["tau"] = np.stack([column(f"tau_{i}") for i in range(JOINTS)], axis=-1)

    order = np.argsort(data["t"])
    for key, value in list(data.items()):
        if value is not None:
            data[key] = value[order]

    data["t"] = data["t"] - data["t"][0]
    return data


def _slice_by_time(data: dict[str, np.ndarray], start_s: float, duration_s: float | None) -> dict[str, np.ndarray]:
    end_s = np.inf if duration_s is None else start_s + duration_s
    mask = (data["t"] >= start_s) & (data["t"] <= end_s)
    sliced = {}
    for key, value in data.items():
        sliced[key] = None if value is None else value[mask]
    if sliced["t"].size:
        sliced["t"] = sliced["t"] - sliced["t"][0]
    return sliced


def _downsample(data: dict[str, np.ndarray], max_points: int) -> dict[str, np.ndarray]:
    if max_points <= 0 or data["t"].size <= max_points:
        return data
    stride = int(np.ceil(data["t"].size / max_points))
    downsampled = {}
    for key, value in data.items():
        downsampled[key] = None if value is None else value[::stride]
    return downsampled


def _write_plot(data: dict[str, np.ndarray], csv_path: Path, output_path: Path, title_prefix: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        ("Commanded position", "cmd", "rad"),
        ("Measured position", "q", "rad"),
        ("Measured velocity", "dq", "rad/s"),
    ]
    if data["tau"] is not None:
        rows.append(("Measured torque", "tau", "Nm"))

    fig, axes = plt.subplots(len(rows), 1, figsize=(14, 2.7 * len(rows)), sharex=True)
    if len(rows) == 1:
        axes = [axes]

    colors = ["tab:blue", "tab:orange", "tab:green"]
    for ax, (label, key, unit) in zip(axes, rows):
        values = data[key]
        for joint_index in range(JOINTS):
            ax.plot(
                data["t"],
                values[:, joint_index],
                color=colors[joint_index],
                linewidth=0.9,
                label=f"joint {joint_index}",
            )
        ax.set_ylabel(f"{label}\n({unit})")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8, ncols=3)

    axes[-1].set_xlabel("time (s)")
    duration = float(data["t"][-1] - data["t"][0]) if data["t"].size > 1 else 0.0
    prefix = f"{title_prefix} " if title_prefix else ""
    fig.suptitle(f"{prefix}{csv_path.name} ({duration:.2f} s shown)")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
