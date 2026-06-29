"""Check rollout return targets against discounted cumulative deltas."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tpi_jepa.dataset import TPIRolloutDataset
from tpi_jepa.labels import DEFAULT_LABELS, load_labels
from tpi_jepa.train import discounted_return_targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=100.0)
    parser.add_argument("--max-samples", type=int, default=256)
    args = parser.parse_args()

    rows = load_labels(args.labels)
    dataset = TPIRolloutDataset(rows, max_specs=args.max_samples, max_horizon=5)
    checked = 0
    for sample in dataset:
        targets = discounted_return_targets(sample.delta_fault_coverages, gamma=args.gamma, scale=args.scale)
        running = torch.zeros((), dtype=sample.delta_fault_coverages[0].dtype)
        manual = []
        for value in reversed(sample.delta_fault_coverages):
            running = value * float(args.scale) + float(args.gamma) * running
            manual.append(running)
        manual.reverse()
        for got, want in zip(targets, manual):
            if not torch.allclose(got, want):
                raise SystemExit(f"return target mismatch: got={float(got):.8f} want={float(want):.8f}")
        checked += 1
    print(f"checked_rollouts={checked}")
    print("return_targets=discounted_cumulative")


if __name__ == "__main__":
    main()
