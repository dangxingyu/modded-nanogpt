#!/usr/bin/env python3
"""Recover unreadable PSGD rows from exact persisted terminal trial logs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from records.track_3_optimization.batch_size_cd import (
    monitor_final_recipe_ablation as monitor,
)
from records.track_3_optimization.batch_size_cd.psgdh_merge_recovery import (
    is_complete,
    load_rows,
    write_rows,
)


def recover_rows(
    rows: list[dict[str, Any]],
    trial_evidence: list[tuple[str, dict[str, dict[str, Any]]]],
) -> list[dict[str, Any]]:
    """Patch only incomplete rows backed by an exact persisted terminal marker."""

    evidence_by_case: dict[str, tuple[str, dict[str, Any]]] = {}
    for trial_id, cases in trial_evidence:
        for case_id, evidence in cases.items():
            evidence_by_case.setdefault(str(case_id), (trial_id, evidence))

    recovered: list[dict[str, Any]] = []
    for original in rows:
        row = dict(original)
        if is_complete(row):
            recovered.append(row)
            continue
        case_id = str(row.get("case_id", ""))
        item = evidence_by_case.get(case_id)
        if item is None:
            recovered.append(row)
            continue
        trial_id, evidence = item
        step = int(evidence["step"])
        total = int(evidence["total"])
        expected = int(row.get("train_steps") or total)
        loss = float(evidence["terminal_val_loss"])
        if (
            evidence.get("evidence") != "case_end_done_exit0_persisted1"
            or step != total
            or total != expected
            or loss != loss
        ):
            recovered.append(row)
            continue
        row.update(
            {
                "status": "DONE",
                "failure_kind": "",
                "last_step": step,
                "total_steps": total,
                "last_val_step": step,
                "last_val_loss": loss,
                "best_val_loss": loss,
                "num_val_points": max(int(row.get("num_val_points") or 0), 1),
                "trial_id": trial_id,
                "log_source": "merlin_strict_terminal",
                "terminal_evidence": evidence["evidence"],
                "terminal_evidence_host": evidence.get("host", ""),
            }
        )
        recovered.append(row)
    return recovered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collected", type=Path, required=True)
    parser.add_argument("--job-run-id", required=True)
    parser.add_argument("--trial-id", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write recovered terminal rows while leaving genuinely unfinished cases intact.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evidence = []
    for trial_id in args.trial_id:
        summary = monitor.summarize_logs(args.job_run_id, trial_id)
        evidence.append((trial_id, summary["strict_terminal_cases"]))
    rows = recover_rows(load_rows(args.collected), evidence)
    incomplete = [str(row.get("case_id", "")) for row in rows if not is_complete(row)]
    if incomplete and not args.allow_incomplete:
        raise SystemExit(f"incomplete rows after strict recovery: {incomplete}")
    write_rows(rows, args.output)
    print(f"recovered_rows={sum(row.get('log_source') == 'merlin_strict_terminal' for row in rows)}")
    print(f"wrote {args.output}")
    print(f"wrote {args.output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
