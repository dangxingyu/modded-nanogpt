#!/usr/bin/env python3
"""Reproduce the merged PR #316 PSGD baseline in one packed Merlin job.

Each worker runs the original 512K, 3400-step source without materialization
changes.  Repetitions are parallel worker roles inside one job so the launch
overhead is paid only once.
"""

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
RESULTS = BASE / "results/core_cd/psgdh_pr316_baseline"
PAYLOADS = BASE / "payloads/psgdh_pr316_baseline"
PR316_BATCH_SIZE = 8 * 64 * 1024
PR316_TRAIN_STEPS = 3400
PR316_REFERENCE_LOSSES = (3.2764, 3.2777, 3.2774, 3.2750, 3.2768)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stamp",
        default=f"pretraining_psgdh_pr316_baseline_{time.strftime('%Y%m%d_%H%M')}",
    )
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--control-plane", default="cn-seed")
    parser.add_argument("--group-id", type=int, default=2089)
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
    parser.add_argument("--image-url", default=core.DEFAULT_IMAGE)
    parser.add_argument("--repo-mnt", default="/opt/tiger/modded-nanogpt")
    parser.add_argument(
        "--track3-data-hdfs",
        default=f"{core.HDFS_BASE}/modded-nanogpt/data/fineweb10B",
    )
    parser.add_argument("--output-base", default=core.DEFAULT_OUTPUT_BASE)
    parser.add_argument(
        "--hdfs-code-dir", default=f"{core.HDFS_BASE}/modded-nanogpt"
    )
    parser.add_argument("--skip-package", action="store_true")
    parser.add_argument("--hdfs-code-tgz", default="")
    parser.add_argument("--apply-bad-host-mask", action="store_true", default=True)
    parser.add_argument("--shampoo-sync-fences", action="store_false", default=False)
    return parser.parse_args()


def build_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.workers < 4:
        raise SystemExit("Packed baseline must request at least 32 H20 GPUs")
    cases: list[dict[str, Any]] = []
    for index in range(args.workers):
        case_id = f"{index:04d}_psgdh_pr316_exact_replicate_{index + 1}"
        env = dict(anchor.BASE_ENV)
        env.update(
            {
                "TRACK3_STAMP": args.stamp,
                "TRACK3_CASE_ID": case_id,
                "TRACK3_RECIPE": "psgdh_pr316_exact",
                "TRACK3_BATCH_SIZE": str(PR316_BATCH_SIZE),
                "TRACK3_BATCH_RATIO": "1",
                "TRACK3_TRAIN_STEPS": str(PR316_TRAIN_STEPS),
                "TRACK3_COORD": "baseline_replicate",
                "TRACK3_BASE_LR_SCALE": "1",
                "TRACK3_SEED": str(index + 1),
                "KL_SOAP_SEED": str(index + 1),
                "TRACK3_DATA_HDFS": args.track3_data_hdfs,
                "TRACK3_OUTPUT_BASE": args.output_base,
                "TRACK3_HARDWARE_FAMILY": "H20",
                "TRACK3_DIST_TIMEOUT_MINUTES": "180",
                "TRACK3_CASE_STARTUP_TIMEOUT_SECONDS": "1200",
                "TRACK3_CASE_STALL_TIMEOUT_SECONDS": "1800",
                "TRACK3_CASE_MAX_ATTEMPTS_PER_WORKER": "3",
                "NPROC": "8",
            }
        )
        cases.append(
            {
                "case_index": index,
                "case_id": case_id,
                "kind": "psgdh_pr316_exact_baseline_reproduction",
                "family": "hyperball",
                "recipe": "psgdh_pr316_exact",
                "label": "PSGD-H PR316 exact",
                "batch": "512k",
                "train_steps": PR316_TRAIN_STEPS,
                "coord": "baseline_replicate",
                "value": index + 1,
                "anchor_value": None,
                "historical_anchor_loss": sum(PR316_REFERENCE_LOSSES)
                / len(PR316_REFERENCE_LOSSES),
                "env": env,
            }
        )
    return cases


def schedule_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    queues = []
    for index, case in enumerate(cases):
        queued = copy.deepcopy(case)
        queued["estimated_runtime_minutes"] = 25.0
        queued["estimated_h20_minutes"] = 25.0
        queues.append(
            {
                "worker_index": index,
                "estimated_runtime_minutes": 25.0,
                "estimated_h20_minutes": 25.0,
                "cases": [queued],
            }
        )
    return {
        "schema": "track3_psgdh_pr316_baseline_schedule_v1",
        "stamp": cases[0]["env"]["TRACK3_STAMP"],
        "campaign_stamp": cases[0]["env"]["TRACK3_STAMP"],
        "batch": "512k",
        "output_base": cases[0]["env"]["TRACK3_OUTPUT_BASE"],
        "worker_count": len(queues),
        "gpus_per_worker": 8,
        "gpuv": "NVIDIA_H20",
        "runtime_label": "h20",
        "case_count": len(cases),
        "worker_queues": queues,
        "reference": {
            "hardware": "NVIDIA H100 80GB HBM3",
            "losses": list(PR316_REFERENCE_LOSSES),
            "mean_loss": sum(PR316_REFERENCE_LOSSES) / len(PR316_REFERENCE_LOSSES),
        },
    }


def main() -> None:
    args = parse_args()
    if args.submit and args.dry_run:
        raise SystemExit("--submit and --dry-run are mutually exclusive")
    if not args.stamp.startswith("pretraining_"):
        raise SystemExit("Use a generic pretraining_* campaign stamp")
    cases = build_cases(args)
    schedule = schedule_cases(cases)
    code_tgz = packed.package_code(args)
    if args.submit:
        packed.verify_hdfs_code(code_tgz)

    result_dir = RESULTS / args.stamp
    payload_dir = PAYLOADS / args.stamp
    schedule_path = result_dir / "schedule.json"
    manifest_path = result_dir / "manifest.json"
    payload_path = payload_dir / f"packed_{args.workers}workers_{args.workers * 8}h20.json"
    receipt_path = result_dir / "launch_receipt.json"
    packed.write_json(schedule_path, schedule)
    packed.write_json(manifest_path, cases)
    payload = packed.build_payload(
        args, "512k", args.workers, schedule_path.read_bytes(), code_tgz
    )
    payload["caption"] = f"m14.2-pretraining-psgdh-pr316-{args.stamp[-13:]}"[:90]
    payload["comment"] = (
        "Exact unmodified PR316 PSGD baseline reproduction; "
        f"replicates={len(cases)} workers={args.workers}; packed"
    )
    payload["tags"] = ["pretraining", "optimization", "psgd", "baseline", args.stamp]
    packed.write_json(payload_path, payload)

    summary = {
        "stamp": args.stamp,
        "cases": len(cases),
        "workers": args.workers,
        "requested_h20": args.workers * 8,
        "estimated_makespan_minutes": 25.0,
        "schedule": str(schedule_path),
        "manifest": str(manifest_path),
        "payload": str(payload_path),
        "hdfs_code_tgz": code_tgz,
        "reference_mean_loss_h100": schedule["reference"]["mean_loss"],
    }
    print(json.dumps(summary, indent=2), flush=True)
    if not args.submit:
        return
    live = packed.live_resource_check(args, args.workers * 8)
    launch = packed.submit_payload(args, payload_path)
    receipt = {
        **summary,
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "live_resource_check": live,
        **launch,
    }
    packed.write_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
