#!/usr/bin/env python3
"""Pack unresolved cases from one Track-3 PSGD-H coordinate round."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

from records.track_3_optimization.batch_size_cd import final_recipe_ablation_campaign as packed
from records.track_3_optimization.batch_size_cd import psgdh_anchor_campaign as anchor
from records.track_3_optimization.batch_size_cd import submit_core_hparam_cd as core


BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results/core_cd/psgdh_anchor"
PAYLOADS = BASE / "payloads/psgdh_anchor"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--collected", type=Path, required=True)
    parser.add_argument(
        "--stamp",
        default=f"pretraining_psgdh_pcd_recovery_{time.strftime('%Y%m%d_%H%M')}",
    )
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--case-stall-timeout-seconds", type=int, default=300)
    parser.add_argument("--control-plane", default="cn-seed")
    parser.add_argument("--group-id", type=int, default=2090)
    parser.add_argument("--cluster-id", type=int, default=47)
    parser.add_argument(
        "--queue-name",
        default="nvidia-h20.hpccluster-yecd0gve6pioxhtqvap6.ai",
    )
    parser.add_argument("--gpuv", default="NVIDIA_H20")
    parser.add_argument("--queue-priority", type=int, default=50)
    parser.add_argument("--retry-times", type=int, default=1)
    parser.add_argument("--role-cpu", type=int, default=112)
    parser.add_argument("--role-memory", type=int, default=1859584)
    parser.add_argument("--use-unguaranteed-quota", action="store_true")
    parser.add_argument("--allow-unlisted-resource", action="store_true")
    parser.add_argument("--image-url", default=core.DEFAULT_IMAGE)
    parser.add_argument("--repo-mnt", default="/opt/tiger/modded-nanogpt")
    parser.add_argument("--output-base", default=core.DEFAULT_OUTPUT_BASE)
    parser.add_argument(
        "--hdfs-code-dir", default=f"{core.HDFS_BASE}/modded-nanogpt"
    )
    parser.add_argument("--skip-package", action="store_true")
    parser.add_argument("--hdfs-code-tgz", default="")
    parser.add_argument("--apply-bad-host-mask", action="store_true", default=True)
    parser.add_argument("--shampoo-sync-fences", action="store_false", default=False)
    return parser.parse_args()


def is_resolved(row: dict[str, Any]) -> bool:
    """Match the terminal contract enforced by ``psgdh_cd_select``."""

    return (
        row.get("status") == "DONE"
        and row.get("last_val_loss") is not None
        and int(row.get("last_step") or -1) == int(row.get("total_steps") or -2)
    )


def unresolved_cases(
    manifest: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    stamp: str,
    case_stall_timeout_seconds: int = 300,
) -> list[dict[str, Any]]:
    if case_stall_timeout_seconds < 1:
        raise ValueError("case_stall_timeout_seconds must be positive")
    resolved = {str(row["case_id"]) for row in rows if is_resolved(row)}
    cases = []
    for original in manifest:
        if str(original["case_id"]) in resolved:
            continue
        case = copy.deepcopy(original)
        case["env"]["TRACK3_STAMP"] = stamp
        case["env"]["TRACK3_CASE_STALL_TIMEOUT_SECONDS"] = str(
            case_stall_timeout_seconds
        )
        # Preserve case_id and every scientific environment value so the
        # source and recovery rows can be merged without changing identity.
        cases.append(case)
    return cases


def schedule_recovery_cases(
    cases: list[dict[str, Any]], workers: int
) -> dict[str, Any]:
    """Create a 32-GPU-minimum schedule, allowing idle recovery lanes."""

    workers = max(4, workers)
    queues = [
        {
            "worker_index": index,
            "estimated_runtime_minutes": 0.0,
            "estimated_h20_minutes": 0.0,
            "cases": [],
        }
        for index in range(workers)
    ]
    for case in cases:
        queue = min(
            queues,
            key=lambda item: (item["estimated_runtime_minutes"], item["worker_index"]),
        )
        queued = copy.deepcopy(case)
        runtime_minutes = max(
            30.0,
            12.0 * float(case["train_steps"]) / 3250.0,
        )
        queued["estimated_runtime_minutes"] = runtime_minutes
        queued["estimated_h20_minutes"] = runtime_minutes
        queue["cases"].append(queued)
        queue["estimated_runtime_minutes"] += runtime_minutes
        queue["estimated_h20_minutes"] += runtime_minutes
    first = cases[0]
    return {
        "schema": "track3_psgdh_pcd_schedule_v2",
        "stamp": first["env"]["TRACK3_STAMP"],
        "campaign_stamp": first["env"]["TRACK3_STAMP"],
        "batch": first["batch"],
        "output_base": first["env"]["TRACK3_OUTPUT_BASE"],
        "worker_count": workers,
        "gpus_per_worker": 8,
        "gpuv": "NVIDIA_H20",
        "runtime_label": "h20",
        "case_count": len(cases),
        "worker_queues": queues,
    }


def main() -> None:
    args = parse_args()
    if args.submit and args.dry_run:
        raise SystemExit("--submit and --dry-run are mutually exclusive")
    if not args.stamp.startswith("pretraining_"):
        raise SystemExit("Use a generic pretraining_* campaign stamp")
    manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    rows = json.loads(args.collected.read_text(encoding="utf-8"))
    cases = unresolved_cases(
        manifest,
        rows,
        args.stamp,
        case_stall_timeout_seconds=args.case_stall_timeout_seconds,
    )
    if not cases:
        print(json.dumps({"stamp": args.stamp, "cases": 0, "complete": True}, indent=2))
        return
    args.batch = str(cases[0]["batch"])
    if any(str(case["batch"]) != args.batch for case in cases):
        raise SystemExit("A packed recovery job must contain exactly one batch size")
    workers = max(4, min(args.workers, len(cases)))
    schedule = schedule_recovery_cases(cases, workers)
    code_tgz = packed.package_code(args)
    if args.submit:
        packed.verify_hdfs_code(code_tgz)

    result_dir = RESULTS / args.stamp
    payload_dir = PAYLOADS / args.stamp
    schedule_path = result_dir / "schedule.json"
    manifest_path = result_dir / "manifest.json"
    payload_path = payload_dir / f"packed_{workers}workers_{workers * 8}h20.json"
    receipt_path = result_dir / "launch_receipt.json"
    packed.write_json(schedule_path, schedule)
    packed.write_json(manifest_path, cases)
    payload = packed.build_payload(
        args, args.batch, workers, schedule_path.read_bytes(), code_tgz
    )
    anchor.apply_quota_mode(payload, args.use_unguaranteed_quota)
    payload["caption"] = f"m14.2-pretraining-psgdh-recovery-{args.batch}-{args.stamp[-13:]}"[:90]
    payload["comment"] = (
        f"Track-3 PSGD-H {args.batch} unresolved-case recovery; "
        f"cases={len(cases)} workers={workers}; packed"
    )
    payload["tags"] = ["pretraining", "optimization", "psgd", "recovery", args.stamp]
    packed.write_json(payload_path, payload)

    summary = {
        "stamp": args.stamp,
        "batch": args.batch,
        "cases": len(cases),
        "workers": workers,
        "requested_h20": workers * 8,
        "source_manifest": str(args.source_manifest),
        "collected": str(args.collected),
        "schedule": str(schedule_path),
        "manifest": str(manifest_path),
        "payload": str(payload_path),
        "hdfs_code_tgz": code_tgz,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if not args.submit:
        return
    live = anchor.live_resource_evidence(args, workers * 8)
    launch = packed.submit_payload(args, payload_path)
    packed.write_json(
        receipt_path,
        {
            **summary,
            "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "live_resource_check": live,
            **launch,
        },
    )


if __name__ == "__main__":
    main()
