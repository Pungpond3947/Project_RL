"""Inspect one-leg UAN hardware CSV logs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

UAN_DIR = Path(__file__).resolve().parents[2] / "source" / "physical_locomotion_ai" / "physical_locomotion_ai" / "uan"
sys.path.insert(0, str(UAN_DIR))
from dataset import UANHardwareDataset  # noqa: E402


DEFAULT_CSVS = (
    "uan_sine_20260427_142543.csv",
    "uan_square_20260427_142913.csv",
    "uan_gaussian_20260427_143236.csv",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="*", default=DEFAULT_CSVS, help="CSV files to inspect.")
    parser.add_argument("--episode_steps", type=int, default=1000, help="Rollout length used for valid-start count.")
    args = parser.parse_args()

    dataset = UANHardwareDataset.load(args.csv, episode_steps=args.episode_steps, device="cpu")
    print(f"rows: {dataset.num_rows}")
    print(f"valid starts for {args.episode_steps} steps: {dataset.num_valid_starts}")
    print(f"median dt: {dataset.dt_median:.6f} s")
    for path, length in zip(dataset.source_files, dataset.source_lengths):
        print(f"  {path}: {length} rows")

    for name, values in (("q", dataset.q), ("dq", dataset.dq), ("cmd", dataset.cmd)):
        mins = torch.min(values, dim=0).values.tolist()
        maxs = torch.max(values, dim=0).values.tolist()
        mins_text = ", ".join(f"{v:.4f}" for v in mins)
        maxs_text = ", ".join(f"{v:.4f}" for v in maxs)
        print(f"{name} min: [{mins_text}]")
        print(f"{name} max: [{maxs_text}]")

    if dataset.tau is None:
        print("tau columns: not found")
    else:
        mins = torch.min(dataset.tau, dim=0).values.tolist()
        maxs = torch.max(dataset.tau, dim=0).values.tolist()
        mins_text = ", ".join(f"{v:.4f}" for v in mins)
        maxs_text = ", ".join(f"{v:.4f}" for v in maxs)
        print(f"tau min: [{mins_text}]")
        print(f"tau max: [{maxs_text}]")


if __name__ == "__main__":
    main()
