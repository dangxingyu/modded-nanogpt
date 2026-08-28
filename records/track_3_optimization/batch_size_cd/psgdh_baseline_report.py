#!/usr/bin/env python3
"""Validate and report the exact five-seed PR #316 PSGD reproduction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from records.track_3_optimization.batch_size_cd.psgdh_pr316_baseline_campaign import (
    PR316_REFERENCE_LOSSES,
    PR316_TRAIN_STEPS,
)


SOURCE_RELATIVE = Path("records/track_3_optimization/results/20260527_psgd/train_psgd.py")
SOURCE = Path(__file__).resolve().parents[3] / SOURCE_RELATIVE


def summarize(rows: list[dict[str, Any]], tolerance: float = 0.003) -> dict[str, Any]:
    by_seed = {int(row["seed"]): row for row in rows}
    if set(by_seed) != set(range(1, 6)) or len(rows) != 5:
        raise ValueError("baseline report requires exactly seeds 1 through 5")
    invalid = [
        seed
        for seed, row in by_seed.items()
        if row.get("status") != "DONE"
        or int(row.get("last_step") or -1) != PR316_TRAIN_STEPS
        or int(row.get("total_steps") or -1) != PR316_TRAIN_STEPS
        or not math.isfinite(float(row.get("last_val_loss", math.nan)))
    ]
    if invalid:
        raise ValueError(f"baseline seeds lack full finite terminals: {invalid}")
    losses = [float(by_seed[seed]["last_val_loss"]) for seed in range(1, 6)]
    reference = list(PR316_REFERENCE_LOSSES)
    mean_loss = statistics.mean(losses)
    reference_mean = statistics.mean(reference)
    return {
        "schema": "track3_psgdh_pr316_baseline_report_v1",
        "passed": abs(mean_loss - reference_mean) <= tolerance,
        "acceptance_tolerance": tolerance,
        "hardware": sorted({str(row.get("runtime_device_type", "")) for row in rows}),
        "train_steps": PR316_TRAIN_STEPS,
        "losses": losses,
        "mean_loss": mean_loss,
        "sample_standard_deviation": statistics.stdev(losses),
        "reference_h100_losses": reference,
        "reference_h100_mean_loss": reference_mean,
        "mean_delta_vs_reference": mean_loss - reference_mean,
        "source": SOURCE_RELATIVE.as_posix(),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Exact PSGD PR #316 baseline reproduction",
        "",
        "All five runs use the unmodified merged PR #316 recipe for 3,400 steps.",
        "The gate passes when the H20 mean is within 0.003 loss of the published H100 mean.",
        "",
        "| Seed | H20 loss | Historical H100 loss |",
        "|---:|---:|---:|",
    ]
    for seed, (loss, reference) in enumerate(
        zip(report["losses"], report["reference_h100_losses"], strict=True), start=1
    ):
        lines.append(f"| {seed} | {loss:.5f} | {reference:.5f} |")
    lines.extend(
        [
            "",
            f"H20 mean: {report['mean_loss']:.6f}",
            "",
            f"Historical H100 mean: {report['reference_h100_mean_loss']:.6f}",
            "",
            f"Mean difference: {report['mean_delta_vs_reference']:+.6f}",
            "",
            f"Gate: {'PASS' if report['passed'] else 'FAIL'}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collected", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=0.003)
    args = parser.parse_args()
    report = summarize(json.loads(args.collected.read_text()), args.tolerance)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.output_markdown.write_text(markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
