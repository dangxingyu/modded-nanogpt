#!/usr/bin/env python3
"""Merge a base PSGD-H collection with one or more packed recoveries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(
        isinstance(row, dict) for row in payload
    ):
        raise ValueError(f"{path}: expected a JSON list of row objects")
    return payload


def is_complete(row: dict[str, Any]) -> bool:
    try:
        return (
            row.get("status") == "DONE"
            and int(row["last_step"]) == int(row["total_steps"])
            and int(row["total_steps"]) > 0
            and row.get("last_val_loss") is not None
        )
    except (KeyError, TypeError, ValueError):
        return False


def merge_rows(
    manifest: list[dict[str, Any]],
    collections: list[list[dict[str, Any]]],
    *,
    require_complete: bool = True,
) -> list[dict[str, Any]]:
    expected = [str(case["case_id"]) for case in manifest]
    if len(expected) != len(set(expected)):
        raise ValueError("manifest contains duplicate case_id values")
    expected_set = set(expected)
    selected: dict[str, dict[str, Any]] = {}
    for rows in collections:
        for row in rows:
            case_id = str(row.get("case_id", ""))
            if case_id not in expected_set:
                continue
            current = selected.get(case_id)
            if current is None or (is_complete(row) and not is_complete(current)):
                selected[case_id] = row
    missing = [case_id for case_id in expected if case_id not in selected]
    if missing:
        raise ValueError(f"missing collected rows: {missing}")
    incomplete = [case_id for case_id in expected if not is_complete(selected[case_id])]
    if incomplete and require_complete:
        raise ValueError(f"incomplete collected rows: {incomplete}")
    return [selected[case_id] for case_id in expected]


def write_rows(rows: list[dict[str, Any]], output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    output_csv = output_json.with_suffix(".csv")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--collected", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write the best row per case even when recovery is still needed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, list):
        raise SystemExit("manifest must be a JSON case list")
    rows = merge_rows(
        manifest,
        [load_rows(path) for path in args.collected],
        require_complete=not args.allow_incomplete,
    )
    write_rows(rows, args.output)
    print(f"merged_rows={len(rows)}")
    print(f"wrote {args.output}")
    print(f"wrote {args.output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
