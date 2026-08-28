#!/usr/bin/env python3
"""Select one complete Track-3 PSGD-H parallel-coordinate round."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


COORD_ENV = {
    "matrix_lr": ("TRACK3_MATRIX_LR_MULT", "matrix_lr_mult"),
    "precond_lr": ("TRACK3_PRECOND_LR_MULT", "precond_lr_mult"),
    "matrix_beta1_om": ("TRACK3_MATRIX_BETA1_OM_MULT", "matrix_beta1_om_mult"),
    "aux_lr_global": ("TRACK3_AUX_LR_MULT", "aux_lr_mult"),
    "aux_beta1_om": ("TRACK3_AUX_BETA1_OM_MULT", "aux_beta1_om_mult"),
    "aux_beta2_om": ("TRACK3_AUX_BETA2_OM_MULT", "aux_beta2_om_mult"),
    "aux_cooldown_frac": ("TRACK3_AUX_COOLDOWN_FRAC", "aux_cooldown_frac"),
    "matrix_cooldown_frac": ("TRACK3_H_COOLDOWN_FRAC", "hidden_cooldown_frac"),
}


def finite_loss_or_infinity(row: dict[str, Any]) -> float:
    value = float(row["last_val_loss"])
    return value if math.isfinite(value) else math.inf


def select_round(manifest: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    cases = {case["case_id"]: case for case in manifest}
    observed = {row["case_id"]: row for row in rows if row["case_id"] in cases}
    if set(observed) != set(cases):
        raise ValueError(f"round is incomplete: {len(observed)}/{len(cases)} cases")
    invalid = [
        row["case_id"]
        for row in observed.values()
        if row.get("status") != "DONE"
        or int(row.get("last_step") or -1) != int(row.get("total_steps") or -2)
        or row.get("last_val_loss") is None
    ]
    if invalid:
        raise ValueError(f"round has invalid terminals: {invalid}")
    center_case = next(case for case in manifest if case["coord"] == "center")
    center = observed[center_case["case_id"]]
    if not math.isfinite(float(center["last_val_loss"])):
        raise ValueError("round center has a non-finite terminal loss")
    center_env = {
        key: str(center_case["env"][key]) for key, _ in COORD_ENV.values()
    }
    evidence: dict[str, Any] = {}
    for coordinate, (env_key, column) in COORD_ENV.items():
        candidates = [center_case] + [case for case in manifest if case["coord"] == coordinate]
        winner_case = min(
            candidates,
            key=lambda case: (
                finite_loss_or_infinity(observed[case["case_id"]]),
                case["case_id"],
            ),
        )
        winner = observed[winner_case["case_id"]]
        center_env[env_key] = str(winner_case["env"][env_key])
        values = [float(case["env"][env_key]) for case in candidates]
        selected = float(winner_case["env"][env_key])
        evidence[coordinate] = {
            "winner_case_id": winner_case["case_id"],
            "winner_loss": float(winner["last_val_loss"]),
            "center_loss": float(center["last_val_loss"]),
            "improvement": float(center["last_val_loss"]) - float(winner["last_val_loss"]),
            "selected_value": selected,
            "center_value": float(center_case["env"][env_key]),
            "boundary": winner_case is not center_case
            and (selected == min(values) or selected == max(values)),
            "collector_column": column,
            "scientific_failure_case_ids": [
                case["case_id"]
                for case in candidates
                if not math.isfinite(float(observed[case["case_id"]]["last_val_loss"]))
            ],
        }
    return {
        "schema": "track3_psgdh_cd_selection_v1",
        "stamp": center_case["env"]["TRACK3_STAMP"],
        "batch": center_case["batch"],
        "center_case_id": center_case["case_id"],
        "center_loss": float(center["last_val_loss"]),
        "center_env": center_env,
        "evidence": evidence,
        "converged": all(
            item["winner_case_id"] == center_case["case_id"]
            for item in evidence.values()
        ),
        "needs_boundary_extension": any(item["boundary"] for item in evidence.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--collected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select_round(
        json.loads(args.manifest.read_text(encoding="utf-8")),
        json.loads(args.collected.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
