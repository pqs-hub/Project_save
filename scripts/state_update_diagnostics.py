"""Diagnose whether action-aware state updates create nonzero SCOAP deltas."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tpi_jepa.dataset import TPIDataset
from tpi_jepa.features import SCOAP_END, SCOAP_START
from tpi_jepa.labels import DEFAULT_LABELS, load_labels


def _nonzero_delta_rate(dataset: TPIDataset) -> tuple[int, int]:
    changed = 0
    total = 0
    for idx in range(len(dataset)):
        sample = dataset[idx]
        delta = sample.x_post[:, SCOAP_START:SCOAP_END] - sample.x_pre[:, SCOAP_START:SCOAP_END]
        changed += int(bool((delta.abs().sum() > 1e-8).item()))
        total += 1
    return changed, total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--state-update-mode", default="proxy")
    parser.add_argument("--expect-nonzero-delta-scoap-rate", type=float, default=0.20)
    args = parser.parse_args()

    rows = load_labels(args.labels)
    static_set = TPIDataset(rows, max_specs=args.max_samples, state_update_mode="static")
    proxy_set = TPIDataset(rows, max_specs=args.max_samples, state_update_mode=args.state_update_mode)
    static_changed, static_total = _nonzero_delta_rate(static_set)
    proxy_changed, proxy_total = _nonzero_delta_rate(proxy_set)
    static_rate = static_changed / max(1, static_total)
    proxy_rate = proxy_changed / max(1, proxy_total)
    print(f"static_nonzero_delta_scoap_rate={static_rate:.6f}")
    print(f"{args.state_update_mode}_nonzero_delta_scoap_rate={proxy_rate:.6f}")
    print(f"samples={proxy_total}")
    print(f"finite={bool(torch.isfinite(proxy_set[0].x_post).all().item()) if proxy_total else False}")
    if static_rate != 0.0:
        raise SystemExit("expected static state_update_mode to keep delta_scoap at zero")
    if proxy_rate < float(args.expect_nonzero_delta_scoap_rate):
        raise SystemExit("action-aware state_update_mode produced too few nonzero delta_scoap targets")


if __name__ == "__main__":
    main()
