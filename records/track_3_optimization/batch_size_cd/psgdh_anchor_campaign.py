#!/usr/bin/env python3
"""Launch one packed parallel-coordinate PSGD-H round at one batch size."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any

from records.track_3_optimization.batch_size_cd import final_recipe_ablation_campaign as packed
from records.track_3_optimization.batch_size_cd import submit_core_hparam_cd as core


BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results/core_cd/psgdh_anchor"
PAYLOADS = BASE / "payloads/psgdh_anchor"

BASE_ENV = {
    "TRACK3_MATRIX_LR_MULT": "1",
    "TRACK3_PRECOND_LR_MULT": "1",
    "TRACK3_MATRIX_BETA1_OM_MULT": "1",
    "TRACK3_MATRIX_BETA2_OM_MULT": "1",
    "TRACK3_SHAMPOO_BETA_OM_MULT": "1",
    "TRACK3_AUX_LR_MULT": "1",
    "TRACK3_AUX_BETA1_OM_MULT": "1",
    "TRACK3_AUX_BETA2_OM_MULT": "1",
    "TRACK3_AUX_COOLDOWN_FRAC": "0.5",
    "TRACK3_H_COOLDOWN_FRAC": "1",
    "TRACK3_MATRIX_WD_PEAK": "0",
    "TRACK3_AUX_WD_PEAK": "0",
    "TRACK3_WD_WARMUP_FRAC": "0",
}

COORDINATES = (
    (
        "matrix_lr",
        "TRACK3_MATRIX_LR_MULT",
        (0.5, 2**-0.5, 2**0.5, 2.0),
    ),
    (
        "precond_lr",
        "TRACK3_PRECOND_LR_MULT",
        (0.5, 2**-0.5, 2**0.5, 2.0),
    ),
    (
        "matrix_beta1_om",
        "TRACK3_MATRIX_BETA1_OM_MULT",
        (0.5, 2**-0.5, 2**0.5, 2.0),
    ),
    (
        "aux_lr_global",
        "TRACK3_AUX_LR_MULT",
        (0.5, 2**-0.5, 2**0.5, 2.0),
    ),
    (
        "aux_beta1_om",
        "TRACK3_AUX_BETA1_OM_MULT",
        (0.5, 2**-0.5, 2**0.5, 2.0),
    ),
    (
        "aux_beta2_om",
        "TRACK3_AUX_BETA2_OM_MULT",
        (0.5, 2**-0.5, 2**0.5, 2.0),
    ),
    (
        "aux_cooldown_frac",
        "TRACK3_AUX_COOLDOWN_FRAC",
        (0.0, 0.2, 0.4, 0.6, 0.8),
    ),
    (
        "matrix_cooldown_frac",
        "TRACK3_H_COOLDOWN_FRAC",
        (0.6, 0.8),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stamp",
        default=f"pretraining_psgdh_pcd_{time.strftime('%Y%m%d_%H%M')}",
    )
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch", choices=tuple(core.BATCH_CONFIGS), default="512k")
    parser.add_argument(
        "--center-env",
        action="append",
        default=[],
        help="Override the current center with TRACK3_KEY=value; repeat as needed.",
    )
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
    parser.add_argument(
        "--allow-unlisted-resource",
        action="store_true",
        help="Submit to an explicitly authorized parent group absent from resource listing.",
    )
    parser.add_argument("--image-url", default=core.DEFAULT_IMAGE)
    parser.add_argument("--repo-mnt", default="/opt/tiger/modded-nanogpt")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--token-horizon", type=int, default=core.DEFAULT_TOKEN_HORIZON)
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


def fmt(value: float | int) -> str:
    return f"{float(value):.12g}"


def safe(value: float | int) -> str:
    return fmt(value).replace("-", "m").replace(".", "p")


def build_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    batch = core.BATCH_CONFIGS[args.batch]
    train_steps = math.ceil(args.token_horizon / batch.batch_size)
    specs: list[tuple[str, str | None, float]] = [("center", None, 1.0)]
    for coord, env_key, values in COORDINATES:
        specs.extend((coord, env_key, value) for value in values)

    cases: list[dict[str, Any]] = []
    for index, (coord, env_key, value) in enumerate(specs):
        env = dict(BASE_ENV)
        env.update(core.parse_center_env(args.center_env))
        if env_key is not None:
            env[env_key] = (
                fmt(float(env[env_key]) * value)
                if env_key.endswith("_MULT")
                else fmt(value)
            )
        case_id = f"{index:04d}_psgdh_core_{args.batch}_{coord}_{safe(value)}"
        env.update(
            {
                "TRACK3_STAMP": args.stamp,
                "TRACK3_CASE_ID": case_id,
                "TRACK3_RECIPE": "psgdh_core",
                "TRACK3_BATCH_SIZE": str(batch.batch_size),
                "TRACK3_BATCH_RATIO": fmt(batch.ratio),
                "TRACK3_TRAIN_STEPS": str(train_steps),
                "TRACK3_COORD": coord,
                "TRACK3_BASE_LR_SCALE": "1",
                "TRACK3_SEED": str(args.seed),
                "KL_SOAP_SEED": str(args.seed),
                "TRACK3_DATA_HDFS": args.track3_data_hdfs,
                "TRACK3_OUTPUT_BASE": args.output_base,
                "TRACK3_HARDWARE_FAMILY": "H20",
                "TRACK3_DIST_TIMEOUT_MINUTES": "180",
                "TRACK3_CASE_STARTUP_TIMEOUT_SECONDS": "1200",
                "TRACK3_CASE_STALL_TIMEOUT_SECONDS": "1800",
                "TRACK3_CASE_MAX_ATTEMPTS_PER_WORKER": "5",
                "NPROC": str(batch.nproc),
            }
        )
        cases.append(
            {
                "case_index": index,
                "case_id": case_id,
                "kind": "psgdh_anchor_coordinate_sweep",
                "family": "hyperball",
                "recipe": "psgdh_core",
                "label": "PSGD-H",
                "batch": args.batch,
                "train_steps": train_steps,
                "coord": coord,
                "value": value if env_key is None else float(env[env_key]),
                "factor": value,
                "anchor_value": (
                    None
                    if coord == "center"
                    else float(dict(BASE_ENV, **core.parse_center_env(args.center_env))[env_key])
                ),
                "historical_anchor_loss": 3.29082,
                "env": env,
            }
        )
    return cases


def schedule_cases(
    cases: list[dict[str, Any]], workers: int
) -> dict[str, Any]:
    if workers < 4:
        raise SystemExit("Packed PSGD campaign must request at least 32 H20 GPUs")
    if workers > len(cases):
        raise SystemExit("workers cannot exceed case count")
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
        # Larger global batches use proportionally more accumulation work per
        # optimizer step.  Fixed-token cases therefore retain an observed
        # ~30 minute floor instead of becoming arbitrarily cheap in the LPT
        # estimate.  The 128K step-count term captures its extra loop overhead.
        runtime_minutes = max(
            30.0,
            12.0 * float(case["train_steps"]) / 3250.0,
        )
        queued["estimated_runtime_minutes"] = runtime_minutes
        queued["estimated_h20_minutes"] = runtime_minutes
        queue["cases"].append(queued)
        queue["estimated_runtime_minutes"] += runtime_minutes
        queue["estimated_h20_minutes"] += runtime_minutes
    return {
        "schema": "track3_psgdh_pcd_schedule_v2",
        "stamp": cases[0]["env"]["TRACK3_STAMP"],
        "campaign_stamp": cases[0]["env"]["TRACK3_STAMP"],
        "batch": cases[0]["batch"],
        "output_base": cases[0]["env"]["TRACK3_OUTPUT_BASE"],
        "worker_count": workers,
        "gpus_per_worker": 8,
        "gpuv": "NVIDIA_H20",
        "runtime_label": "h20",
        "case_count": len(cases),
        "worker_queues": queues,
    }


def apply_quota_mode(payload: dict[str, Any], use_unguaranteed_quota: bool) -> None:
    if use_unguaranteed_quota:
        payload["resource_config"]["arnold_config"]["roles"][0]["advanced"] = {
            "is_use_unguaranteed_quota": True
        }


def live_resource_evidence(args: argparse.Namespace, requested_gpus: int) -> dict[str, Any]:
    try:
        return packed.live_resource_check(args, requested_gpus)
    except SystemExit as exc:
        if not args.allow_unlisted_resource:
            raise
        return {
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "group_sid": str(args.group_id),
            "cluster_sid": str(args.cluster_id),
            "queue_name": args.queue_name,
            "gpu_type": packed.resource_gpu_name(args.gpuv),
            "requested_gpus": requested_gpus,
            "capacity_sufficient_at_submit": None,
            "query_scope": "explicitly_authorized_unlisted_parent_group",
            "note": str(exc),
        }


def main() -> None:
    args = parse_args()
    if args.submit and args.dry_run:
        raise SystemExit("--submit and --dry-run are mutually exclusive")
    if not args.stamp.startswith("pretraining_"):
        raise SystemExit("Use a generic pretraining_* campaign stamp")
    cases = build_cases(args)
    schedule = schedule_cases(cases, args.workers)
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
        args, args.batch, args.workers, schedule_path.read_bytes(), code_tgz
    )
    apply_quota_mode(payload, args.use_unguaranteed_quota)
    payload["caption"] = (
        f"m14.2-pretraining-psgdh-pcd-{args.batch}-{args.stamp[-13:]}"
    )[:90]
    payload["comment"] = (
        f"Track-3 PSGD-H PR316 {args.batch} parallel coordinate round; "
        f"cases={len(cases)} workers={args.workers}; packed"
    )
    payload["tags"] = ["pretraining", "optimization", "psgd", args.stamp]
    packed.write_json(payload_path, payload)

    summary = {
        "stamp": args.stamp,
        "cases": len(cases),
        "workers": args.workers,
        "requested_h20": args.workers * 8,
        "estimated_makespan_minutes": max(
            queue["estimated_runtime_minutes"] for queue in schedule["worker_queues"]
        ),
        "schedule": str(schedule_path),
        "manifest": str(manifest_path),
        "payload": str(payload_path),
        "hdfs_code_tgz": code_tgz,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if not args.submit:
        return
    live = live_resource_evidence(args, args.workers * 8)
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
