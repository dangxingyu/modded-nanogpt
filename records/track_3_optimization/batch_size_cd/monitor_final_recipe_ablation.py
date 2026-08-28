#!/usr/bin/env python3
"""Compact lifecycle and worker-progress monitor for a packed ablation campaign."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import re
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

from records.track_3_optimization.batch_size_cd import final_recipe_ablation_campaign as campaign


DISPATCH_RE = re.compile(
    r"^TRACK3_FINAL_ABLATION_DISPATCH role_index=(?P<index>\d+) host=(?P<host>\S+)",
    re.MULTILINE,
)
STEP_RE = re.compile(r"^step:(?P<step>\d+)/(?P<total>\d+)", re.MULTILINE)
CASE_END_RE = re.compile(
    r"^TRACK3_FINAL_ABLATION_CASE_END .*?case_id=(?P<case_id>\S+) "
    r".*?status=(?P<status>\S+) exit_code=(?P<exit_code>-?\d+)",
    re.MULTILINE,
)
CASE_START_RE = re.compile(
    r"^TRACK3_FINAL_ABLATION_CASE_START .*?case_id=(?P<case_id>\S+)",
    re.MULTILINE,
)
STRICT_CASE_END_RE = re.compile(
    r"^TRACK3_FINAL_ABLATION_CASE_END .*?case_id=(?P<case_id>\S+) "
    r".*?status=done exit_code=0 persisted=1(?:\s|$)",
    re.MULTILINE,
)
SKIP_COMPLETE_RE = re.compile(
    r"^TRACK3_FINAL_ABLATION_SKIP_COMPLETE .*?case_id=(?P<case_id>\S+)",
    re.MULTILINE,
)
TERMINAL_VAL_RE = re.compile(
    r"^step:(?P<step>\d+)/(?P<total>\d+)\s+"
    r"val_loss:(?P<loss>[-+0-9.eE]+)",
    re.MULTILINE,
)

COLLECTIVE_TIMEOUT_MARKERS = (
    "Watchdog caught collective operation timeout",
    "ProcessGroupNCCL's watchdog got stuck",
    "Terminating the process after attempting to dump debug info, due to "
    "ProcessGroupNCCL watchdog hang",
)
NCCL_SYSTEM_ERROR_MARKERS = (
    "ncclSystemError",
    "pthread_join failed",
)


def has_collective_timeout(stderr: str) -> bool:
    """Recognize both operation timeouts and NCCL watchdog-heartbeat hangs."""

    return any(marker in stderr for marker in COLLECTIVE_TIMEOUT_MARKERS)


def has_nccl_system_error(stderr: str) -> bool:
    return any(marker in stderr for marker in NCCL_SYSTEM_ERROR_MARKERS)


def strict_terminal_cases_from_stdout(
    stdout: str, host: str
) -> dict[str, dict[str, Any]]:
    """Extract only persisted exact-terminal cases from one sequential lane."""

    starts = list(CASE_START_RE.finditer(stdout))
    strict: dict[str, dict[str, Any]] = {}
    for end in STRICT_CASE_END_RE.finditer(stdout):
        case_id = end.group("case_id")
        matching = [
            start for start in starts
            if start.start() < end.start() and start.group("case_id") == case_id
        ]
        if not matching:
            continue
        start = matching[-1]
        terminals = list(TERMINAL_VAL_RE.finditer(
            stdout, start.end(), end.start()
        ))
        exact = [
            item for item in terminals
            if int(item.group("step")) == int(item.group("total"))
        ]
        if not exact:
            continue
        terminal = exact[-1]
        try:
            loss = float(terminal.group("loss"))
        except ValueError:
            continue
        if not math.isfinite(loss):
            continue
        strict[case_id] = {
            "case_id": case_id,
            "step": int(terminal.group("step")),
            "total": int(terminal.group("total")),
            "terminal_val_loss": loss,
            "host": host,
            "evidence": "case_end_done_exit0_persisted1",
        }
    return strict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--campaign-root", type=Path, default=campaign.DEFAULT_MANIFEST_DIR)
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args()


def merlin_json(command: list[str], payload: dict[str, Any]) -> dict[str, Any]:
    last_error = ""
    for attempt in range(3):
        result = subprocess.run(
            [
                campaign.MERLIN,
                "--control-plane",
                "cn-seed",
                *command,
                "--json",
                json.dumps(payload),
            ],
            cwd=campaign.ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
        last_error = result.stderr or result.stdout
        if attempt < 2:
            time.sleep(attempt + 1)
    raise RuntimeError(last_error)


def active_launches(receipts: dict[str, Any]) -> dict[str, dict[str, Any]]:
    active: dict[str, list[dict[str, Any]]] = {}
    for item in receipts.get("launches", []):
        if not item.get("superseded_at"):
            active.setdefault(str(item["batch"]), []).append(item)
    duplicates = {batch: rows for batch, rows in active.items() if len(rows) != 1}
    if duplicates:
        raise ValueError(f"Expected one active receipt per batch: {duplicates}")
    return {batch: rows[0] for batch, rows in active.items()}


def fetch_log(item: dict[str, Any]) -> tuple[str, str, str]:
    try:
        with urllib.request.urlopen(str(item["url"]), timeout=20) as handle:
            text = handle.read().decode("utf-8", "replace")
        return str(item["pod_name"]), str(item["type"]), text
    except Exception as exc:  # Log proxy failure must not abort lifecycle polling.
        return str(item["pod_name"]), str(item["type"]), f"FETCH_ERROR {type(exc).__name__}: {exc}"


def summarize_logs(job_run_id: str, trial_id: str) -> dict[str, Any]:
    obj = merlin_json(
        ["job", "list-trial-logs"],
        {
            "job_run_id": job_run_id,
            "trial_id": trial_id,
            "filter": {"log_type": "instance_log"},
        },
    )
    items = obj.get("log_list", [])
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as executor:
        fetched = list(executor.map(fetch_log, items))
    pods: dict[str, dict[str, str]] = {}
    for pod, log_type, text in fetched:
        pods.setdefault(pod, {})[log_type] = text

    dispatch: dict[int, str] = {}
    starts = 0
    started_case_ids: set[str] = set()
    ends: list[dict[str, Any]] = []
    latest_end_by_case: dict[str, dict[str, Any]] = {}
    max_steps: list[int] = []
    active_cases: dict[str, str] = {}
    latest_steps: dict[str, int] = {}
    collective_timeout_hosts: list[str] = []
    nccl_system_hosts: list[str] = []
    fetch_errors = 0
    strict_terminal_cases: dict[str, dict[str, Any]] = {}
    skipped_complete_cases: set[str] = set()
    for logs in pods.values():
        stdout = logs.get("stdout", "")
        stderr = logs.get("stderr", "")
        match = DISPATCH_RE.search(stdout)
        host = match.group("host") if match else "unknown"
        if match:
            dispatch[int(match.group("index"))] = host
        strict_terminal_cases.update(
            strict_terminal_cases_from_stdout(stdout, host)
        )
        skipped_complete_cases.update(
            item.group("case_id") for item in SKIP_COMPLETE_RE.finditer(stdout)
        )
        start_matches = list(CASE_START_RE.finditer(stdout))
        end_matches = list(CASE_END_RE.finditer(stdout))
        starts += len(start_matches)
        started_case_ids.update(item.group("case_id") for item in start_matches)
        parsed_ends = [
            {
                "case_id": item.group("case_id"),
                "status": item.group("status"),
                "exit_code": int(item.group("exit_code")),
                "host": host,
            }
            for item in end_matches
        ]
        ends.extend(parsed_ends)
        for parsed_end in parsed_ends:
            latest_end_by_case[str(parsed_end["case_id"])] = parsed_end
        steps = [int(item.group("step")) for item in STEP_RE.finditer(stdout)]
        if steps:
            max_steps.append(max(steps))
        if start_matches and (
            not end_matches or start_matches[-1].start() > end_matches[-1].start()
        ):
            active_cases[host] = start_matches[-1].group("case_id")
            active_steps = [
                int(item.group("step"))
                for item in STEP_RE.finditer(stdout, start_matches[-1].end())
            ]
            if active_steps:
                latest_steps[host] = active_steps[-1]
        if has_collective_timeout(stderr):
            collective_timeout_hosts.append(host)
        if has_nccl_system_error(stderr):
            nccl_system_hosts.append(host)
        fetch_errors += int("FETCH_ERROR " in stdout) + int("FETCH_ERROR " in stderr)
    active_case_ids = set(active_cases.values())
    unresolved_failed_cases = [
        item
        for case_id, item in latest_end_by_case.items()
        if item["status"] == "failed" and case_id not in active_case_ids
    ]
    return {
        "stdout_pods": sum("stdout" in logs for logs in pods.values()),
        "stderr_pods": sum("stderr" in logs for logs in pods.values()),
        "dispatch": {str(index): host for index, host in sorted(dispatch.items())},
        "dispatch_unique": len(dispatch),
        "case_starts": starts,
        "unique_cases_started": len(started_case_ids),
        "retry_attempts_started": starts - len(started_case_ids),
        "case_ends": len(ends),
        "case_done": sum(item["status"] == "done" for item in ends),
        "case_failed": sum(item["status"] == "failed" for item in ends),
        "done_cases": sorted(
            item["case_id"] for item in ends if item["status"] == "done"
        ),
        "resolved_case_done": sum(
            item["status"] == "done" for item in latest_end_by_case.values()
        ),
        "resolved_done_cases": sorted(
            case_id
            for case_id, item in latest_end_by_case.items()
            if item["status"] == "done"
        ),
        "strict_terminal_cases": strict_terminal_cases,
        "skipped_complete_cases": sorted(skipped_complete_cases),
        "unresolved_failed_cases": unresolved_failed_cases,
        "active_cases": active_cases,
        "latest_steps": latest_steps,
        "failed_cases": [item for item in ends if item["status"] == "failed"],
        "max_step_min": min(max_steps) if max_steps else None,
        "max_step_max": max(max_steps) if max_steps else None,
        "collective_timeout_hosts": sorted(set(collective_timeout_hosts)),
        "nccl_system_hosts": sorted(set(nccl_system_hosts)),
        "fetch_errors": fetch_errors,
    }


def main() -> None:
    args = parse_args()
    campaign_dir = args.campaign_root / args.stamp
    manifest = json.loads((campaign_dir / "campaign_manifest.json").read_text())
    receipts = json.loads((campaign_dir / "launch_receipts.json").read_text())
    launches = active_launches(receipts)
    batches: dict[str, Any] = {}
    now_ms = int(time.time() * 1000)
    for batch in campaign.BATCH_ORDER:
        launch = launches.get(batch)
        if launch is None:
            batches[batch] = {"status": "NO_ACTIVE_RECEIPT"}
            continue
        job_run_id = str(launch["job_run_id"])
        item = merlin_json(
            ["job", "get-run"], {"job_run_id": job_run_id, "timeout": 2}
        )["job_run"]
        trial_id = str(item.get("meta", {}).get("arnold_trial_id") or "")
        started_ms = int(
            item.get("start_running_time")
            or item.get("meta", {}).get("start_running_time")
            or 0
        )
        # A batch can be rehomed or resized after the campaign manifest is
        # frozen.  Report the active launch shape, not the original schedule
        # shape from a superseded receipt.
        launch_hardware = launch.get("hardware_override", {})
        worker_count_expected = int(
            launch_hardware.get(
                "worker_count", manifest["batches"][batch]["worker_count"]
            )
        )
        gpus_expected = int(launch.get("gpus", worker_count_expected * 8))
        row: dict[str, Any] = {
            "job_run_id": job_run_id,
            "status": item.get("status"),
            "trial_id": trial_id or None,
            "worker_count_expected": worker_count_expected,
            "gpus_expected": gpus_expected,
            "running_minutes": (
                round((now_ms - started_ms) / 60000, 2) if started_ms else None
            ),
            "error": item.get("meta", {}).get("err_msg") or None,
        }
        if trial_id and item.get("status") in {"RUNNING", "DONE", "FAILED", "STOPPED"}:
            try:
                row.update(summarize_logs(job_run_id, trial_id))
            except Exception as exc:
                # A transient Merlin log-proxy or endpoint failure must not
                # suppress lifecycle state for every other batch.  Keep the
                # error deliberately terse so signed log URLs or credentials
                # can never leak into the persisted monitor artifact.
                row["log_summary_error"] = (
                    f"{type(exc).__name__}: list-trial-logs unavailable after retries"
                )
            else:
                progress_gate_min_step = 10
                worker_progress = list(row["latest_steps"].values())
                row["progress_gate_min_step"] = progress_gate_min_step
                row["passed_600s_subgroup_gate"] = bool(
                    row["running_minutes"] is not None
                    and row["running_minutes"] >= 12
                    and row["dispatch_unique"] == row["worker_count_expected"]
                    and not row["unresolved_failed_cases"]
                    and not row["collective_timeout_hosts"]
                    and not row["nccl_system_hosts"]
                    and len(worker_progress) == row["worker_count_expected"]
                    and min(worker_progress) >= progress_gate_min_step
                )
        batches[batch] = row

    output = {
        "schema": "track3_final_recipe_ablation_monitor_v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "campaign": args.stamp,
        "batches": batches,
    }
    if not args.no_write:
        path = campaign_dir / "monitor_latest.json"
        path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        output["written_to"] = str(path)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
