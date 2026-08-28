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
BATCH_ORDER = {"128k": 0, "512k": 1, "1m": 2, "2m": 3}


def batch_sort_key(batch: str) -> tuple[int, str]:
    return BATCH_ORDER.get(batch, len(BATCH_ORDER)), batch


def load_round(path: Path) -> dict[str, Any]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    collected_path = next(
        (
            candidate
            for candidate in (
                path / "collected_complete.json",
                path / "collected_merged.json",
                path / "collected_final.json",
                path / "collected.json",
            )
            if candidate.is_file()
        ),
        None,
    )
    if collected_path is None:
        raise ValueError(f"{path}: no collected result file")
    collected = json.loads(collected_path.read_text(encoding="utf-8"))
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
        "collected_path": collected_path,
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
    coordinate_rows = []
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
            for coordinate, evidence in selection["evidence"].items():
                coordinate_rows.append(
                    {
                        "batch": batch,
                        "round_index": index,
                        "coordinate": coordinate,
                        "raw_best_case_id": evidence["raw_best_case_id"],
                        "raw_improvement": evidence["raw_improvement"],
                        "accepted": evidence["winner_case_id"]
                        != selection["center_case_id"],
                        "accepted_improvement": evidence["improvement"],
                        "boundary": evidence["boundary"],
                    }
                )
    round_fields = tuple(round_rows[0])
    with (output_dir / "round_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=round_fields)
        writer.writeheader()
        writer.writerows(round_rows)
    coordinate_fields = tuple(coordinate_rows[0])
    with (output_dir / "coordinate_improvements.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=coordinate_fields)
        writer.writeheader()
        writer.writerows(coordinate_rows)

    ordered_batches = sorted(by_batch, key=batch_sort_key)
    final_recipes = {
        batch: by_batch[batch][-1]["selection"]["center_env"]
        for batch in ordered_batches
    }
    result = {
        "schema": "track3_psgdh_cd_report_v1",
        "acceptance_threshold": 0.003,
        "batches": ordered_batches,
        "round_count": len(rounds),
        "final_recipes": final_recipes,
        "rounds": round_rows,
    }
    (output_dir / "final_recipes.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_plots(round_rows, coordinate_rows, output_dir)
    write_markdown(result, output_dir)
    return result


def write_plots(
    round_rows: list[dict[str, Any]],
    coordinate_rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    by_batch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in round_rows:
        by_batch[row["batch"]].append(row)
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    for batch in sorted(by_batch, key=batch_sort_key):
        rows = by_batch[batch]
        axis.plot(
            [row["round_index"] for row in rows],
            [row["center_loss"] for row in rows],
            "o-",
            label=batch,
        )
    axis.set_xlabel("Coordinate-descent round")
    axis.set_ylabel("Center validation loss")
    axis.legend(title="Batch")
    figure.tight_layout()
    figure.savefig(output_dir / "center_loss_by_round.svg")
    plt.close(figure)

    accepted = [row for row in coordinate_rows if row["accepted"]]
    if not accepted:
        return
    labels = [
        f"{row['batch']} r{row['round_index']} {row['coordinate']}" for row in accepted
    ]
    figure, axis = plt.subplots(figsize=(8.0, max(3.0, 0.32 * len(labels))))
    positions = list(range(len(accepted)))
    axis.barh(positions, [row["accepted_improvement"] for row in accepted])
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Accepted validation-loss improvement")
    figure.tight_layout()
    figure.savefig(output_dir / "accepted_coordinate_improvements.svg")
    plt.close(figure)


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
            "## Final recipes",
            "",
            "Learning-rate entries multiply the corresponding PR #316 reference",
            "learning rate. A first- or second-moment entry multiplies one minus",
            "the corresponding reference beta before the beta is reconstructed.",
            "Cooldown entries are absolute fractions of the training horizon.",
            "",
            "| Batch | Matrix LR factor | Preconditioner LR factor | Matrix first-moment factor | Auxiliary LR factor | Auxiliary first-moment factor | Auxiliary second-moment factor | Auxiliary cooldown | Matrix cooldown |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    keys = (
        "TRACK3_MATRIX_LR_MULT",
        "TRACK3_PRECOND_LR_MULT",
        "TRACK3_MATRIX_BETA1_OM_MULT",
        "TRACK3_AUX_LR_MULT",
        "TRACK3_AUX_BETA1_OM_MULT",
        "TRACK3_AUX_BETA2_OM_MULT",
        "TRACK3_AUX_COOLDOWN_FRAC",
        "TRACK3_H_COOLDOWN_FRAC",
    )
    for batch in result["batches"]:
        recipe = result["final_recipes"][batch]
        values = " | ".join(str(recipe.get(key, "")) for key in keys)
        lines.append(f"| {batch} | {values} |")
    lines.extend(
        [
            "",
            "The complete per-case observations are in `raw_results.csv`. The",
            "round-level decisions are in `round_summary.csv`, every coordinate",
            "decision is in `coordinate_improvements.csv`, and the exact final",
            "environment multipliers are in `final_recipes.json`.",
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
