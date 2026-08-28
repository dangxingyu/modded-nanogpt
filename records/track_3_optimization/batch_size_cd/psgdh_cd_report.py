#!/usr/bin/env python3
"""Build a credential-free report from completed Track-3 PSGD-H CD rounds."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


RAW_FIELDS = (
    "stamp",
    "batch",
    "case_id",
    "coord",
    "coord_value",
    "status",
    "last_step",
    "total_steps",
    "last_val_loss",
    "hardware_family",
    "gpuv",
    "matrix_lr_mult",
    "precond_lr_mult",
    "matrix_beta1_om_mult",
    "aux_lr_mult",
    "aux_beta1_om_mult",
    "aux_beta2_om_mult",
    "aux_cooldown_frac",
    "hidden_cooldown_frac",
)


def load_round(path: Path) -> dict[str, Any]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    collected = json.loads((path / "collected.json").read_text(encoding="utf-8"))
    selection = json.loads((path / "selection.json").read_text(encoding="utf-8"))
    if len(manifest) != len(collected) or len(manifest) != 32:
        raise ValueError(f"{path}: expected one complete 32-case round")
    case_ids = {case["case_id"] for case in manifest}
    if {row["case_id"] for row in collected} != case_ids:
        raise ValueError(f"{path}: collected cases do not match the manifest")
    invalid = [
        row["case_id"]
        for row in collected
        if row.get("status") != "DONE"
        or int(row.get("last_step") or -1) != int(row.get("total_steps") or -2)
        or row.get("last_val_loss") is None
    ]
    if invalid:
        raise ValueError(f"{path}: invalid terminals: {invalid}")
    hardware = {row.get("hardware_family") for row in collected}
    if hardware != {"H20"}:
        raise ValueError(f"{path}: formal CD rows must all use H20, got {hardware}")
    if selection["stamp"] != manifest[0]["env"]["TRACK3_STAMP"]:
        raise ValueError(f"{path}: selection stamp does not match manifest")
    return {
        "path": path,
        "manifest": manifest,
        "collected": collected,
        "selection": selection,
    }


def build_report(round_dirs: list[Path], output_dir: Path) -> dict[str, Any]:
    rounds = [load_round(path) for path in round_dirs]
    by_batch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for round_result in rounds:
        by_batch[round_result["selection"]["batch"]].append(round_result)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = [
        {field: row.get(field) for field in RAW_FIELDS}
        for round_result in rounds
        for row in round_result["collected"]
    ]
    with (output_dir / "raw_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS)
        writer.writeheader()
        writer.writerows(raw_rows)

    round_rows = []
    for batch, batch_rounds in by_batch.items():
        for index, round_result in enumerate(batch_rounds):
            selection = round_result["selection"]
            round_rows.append(
                {
                    "batch": batch,
                    "round_index": index,
                    "stamp": selection["stamp"],
                    "center_loss": selection["center_loss"],
                    "converged": selection["converged"],
                    "needs_boundary_extension": selection["needs_boundary_extension"],
                    "accepted_coordinates": ",".join(
                        coordinate
                        for coordinate, evidence in selection["evidence"].items()
                        if evidence["winner_case_id"] != selection["center_case_id"]
                    ),
                }
            )
    round_fields = tuple(round_rows[0])
    with (output_dir / "round_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=round_fields)
        writer.writeheader()
        writer.writerows(round_rows)

    final_recipes = {
        batch: batch_rounds[-1]["selection"]["center_env"]
        for batch, batch_rounds in sorted(by_batch.items())
    }
    result = {
        "schema": "track3_psgdh_cd_report_v1",
        "acceptance_threshold": 0.003,
        "batches": sorted(by_batch),
        "round_count": len(rounds),
        "final_recipes": final_recipes,
        "rounds": round_rows,
    }
    (output_dir / "final_recipes.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_markdown(result, output_dir)
    return result


def write_markdown(result: dict[str, Any], output_dir: Path) -> None:
    lines = [
        "# Track-3 PSGD-H coordinate descent",
        "",
        "Each round changes one optimizer coordinate at a time around the same",
        "center recipe. All 32 cases in a round use one seed, the same token",
        "horizon, and H20 GPUs. Independent coordinate winners are combined",
        "only when their validation-loss improvement is at least 0.003.",
        "",
        "| Batch | Rounds | Final center loss | Converged |",
        "|---|---:|---:|---|",
    ]
    by_batch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in result["rounds"]:
        by_batch[row["batch"]].append(row)
    for batch in result["batches"]:
        rows = by_batch[batch]
        final = rows[-1]
        lines.append(
            f"| {batch} | {len(rows)} | {final['center_loss']:.6f} | "
            f"{'yes' if final['converged'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "The complete per-case observations are in `raw_results.csv`. The",
            "round-level decisions are in `round_summary.csv`, and the exact",
            "final environment multipliers are in `final_recipes.json`.",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_report(args.round_dir, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
