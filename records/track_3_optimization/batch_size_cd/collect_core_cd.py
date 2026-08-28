#!/usr/bin/env python3
"""Collect core MuonH / KL-SOAP-H coordinate-descent status and losses."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import re
import socket
import statistics
import subprocess
import time
import urllib.error
import urllib.request
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from records.track_3_optimization.batch_size_cd import cd_policy
from records.track_3_optimization.batch_size_cd import submit_standard_wd_cd as standard_cd


ROOT = Path(__file__).resolve().parents[3]
MERLIN = os.environ.get("MERLIN_CLI", "/home/tiger/.merlin-cli/bin/merlin-cli")
OUTPUT_DIR = ROOT / "records/track_3_optimization/batch_size_cd/results/core_cd"
LOCAL_RUN_BASE = Path("/mnt/hdfs/user/xingyu.dang/modded-nanogpt-runs")
HDFS_RUN_BASE = "hdfs://haruna/home/byte_data_seed/hdd_hldy/user/xingyu.dang/modded-nanogpt-runs"
ARNOLD_HDFS_CLI = Path("/opt/tiger/arnold/hdfs_client/hdfs")
HDFS_CLI = os.environ.get(
    "HDFS_CLI",
    str(ARNOLD_HDFS_CLI) if ARNOLD_HDFS_CLI.is_file() else "hdfs",
)
LOG_TAIL_BYTES = 8 * 1024 * 1024
TERMINAL_STATUSES = {"DONE", "FAILED", "FAILED_TO_LAUNCH", "STOPPED"}

STEP_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<total>\d+)(?:\s+val_loss:(?P<val>[+-]?(?:nan|inf|\d+(?:\.\d*)?|\.\d+)))?",
    re.IGNORECASE,
)

RUNTIME_PATTERNS = {
    "runtime_node_spec": r"ARNOLD_NODE_SPEC:\s*([^\s]+)",
    "runtime_node_vendor": r"ARNOLD_NODE_VENDOR:\s*([^\s]+)",
    "runtime_rdma_bigpod": r"ARNOLD_NODE_RDMA_BIGPOD:\s*([^\s]+)",
    "runtime_rdma_minipod": r"ARNOLD_NODE_RDMA_MINIPOD:\s*([^\s]+)",
    "runtime_rdma_switch": r"ARNOLD_NODE_RDMA_SWITCH_NAME:\s*([^\s]+)",
    "runtime_cuda_driver": r"CUDA driver version:\s*([^\s]+)",
    "runtime_cuda_toolkit": r"CUDA toolkit version:\s*([^\s]+)",
    "runtime_device_type": r"ARNOLD_DEVICE_TYPE=([^\s]+)",
}

ENV_KEYS = [
    "TRACK3_MATRIX_LR_MULT",
    "TRACK3_PRECOND_LR_MULT",
    "TRACK3_AUX_LR_MULT",
    "TRACK3_MATRIX_BETA1_OM_MULT",
    "TRACK3_MATRIX_BETA2_OM_MULT",
    "TRACK3_SHAMPOO_BETA_OM_MULT",
    "TRACK3_AUX_BETA1_OM_MULT",
    "TRACK3_AUX_BETA2_OM_MULT",
    "TRACK3_AUX_COOLDOWN_FRAC",
    "TRACK3_MATRIX_WD_PEAK",
    "TRACK3_AUX_WD_PEAK",
    "TRACK3_WD_WARMUP_FRAC",
]

DEFAULT_ENV_VALUES = {
    "TRACK3_MATRIX_LR_MULT": "1",
    "TRACK3_PRECOND_LR_MULT": "1",
    "TRACK3_AUX_LR_MULT": "1",
    "TRACK3_MATRIX_BETA1_OM_MULT": "1",
    "TRACK3_MATRIX_BETA2_OM_MULT": "1",
    "TRACK3_SHAMPOO_BETA_OM_MULT": "1",
    "TRACK3_AUX_BETA1_OM_MULT": "1",
    "TRACK3_AUX_BETA2_OM_MULT": "1",
    "TRACK3_AUX_COOLDOWN_FRAC": "0.4",
    "TRACK3_MATRIX_WD_PEAK": "0",
    "TRACK3_AUX_WD_PEAK": "0",
    "TRACK3_WD_WARMUP_FRAC": "0",
}

RECIPE_DEFAULT_ENV_VALUES = {
    # These recipes inherit train_gpt_simple.py / PR326's WSD cooldown default.
    # Other source records keep the older collector default of 0.4 unless the
    # job explicitly set TRACK3_AUX_COOLDOWN_FRAC.
    "muonw_core": {"TRACK3_AUX_COOLDOWN_FRAC": "0.7"},
    "lionw_core": {"TRACK3_AUX_COOLDOWN_FRAC": "0.7"},
    "lionw_pr326_core": {"TRACK3_AUX_COOLDOWN_FRAC": "0.7"},
    "psgdh_core": {"TRACK3_AUX_COOLDOWN_FRAC": "0.5"},
}

COORD_VALUE_COLUMNS = {
    "matrix_lr": "matrix_lr_mult",
    "precond_lr": "precond_lr_mult",
    "matrix_mu_om": "matrix_beta1_om_mult",
    "matrix_beta1_om": "matrix_beta1_om_mult",
    "matrix_beta2_om": "matrix_beta2_om_mult",
    "shampoo_beta_om": "shampoo_beta_om_mult",
    "aux_lr_global": "aux_lr_mult",
    "aux_beta1_om": "aux_beta1_om_mult",
    "aux_beta2_om": "aux_beta2_om_mult",
    "aux_cooldown_frac": "aux_cooldown_frac",
    "matrix_cooldown_frac": "hidden_cooldown_frac",
    "matrix_wd_peak": "matrix_wd_peak",
    "wd_warmup_frac": "wd_warmup_frac",
}

def run_merlin(command: list[str], payload: dict[str, Any], timeout: int = 45, retries: int = 4) -> dict[str, Any]:
    cmd = [MERLIN, "--control-plane", "cn-seed", *command, "--json", json.dumps(payload)]
    last_error = ""
    for attempt in range(retries):
        try:
            out = subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.STDOUT, timeout=timeout)
            return json.loads(out)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            if isinstance(exc, subprocess.CalledProcessError):
                last_error = str(exc.output or exc)
            elif isinstance(exc, subprocess.TimeoutExpired):
                last_error = f"timeout after {timeout}s"
            else:
                last_error = str(exc)
            if attempt == retries - 1:
                break
            time.sleep(2**attempt)
    raise RuntimeError(f"merlin command failed after retries: {' '.join(cmd)}\n{last_error}")


def stamp_query_terms(stamp: str) -> list[str]:
    query_terms = [stamp]
    no_timestamp = re.sub(r"_\d{8}_\d+$", "", stamp)
    if no_timestamp != stamp:
        query_terms.append(no_timestamp)
    for width in (80, 70, 60, 50):
        if len(stamp) > width:
            query_terms.append(stamp[:width].rstrip("_-"))

    # Recovery and diagnostic payloads use a compact caption to satisfy
    # Merlin's 90-character limit while preserving the full TRACK3_STAMP.
    compact_pattern = r"_(?:seed_us_)?(?:recovery|diagnostic)_wave(\d+)_"
    compact = re.sub(
        compact_pattern,
        r"_w\1_",
        stamp,
        count=1,
    )
    if compact != stamp:
        query_terms.append(compact)
        match = re.search(compact_pattern, stamp)
        if match is not None:
            # Some captions also abbreviate the coordinate suffix (for example
            # matrix_wd_peak -> mwd015). The wave prefix remains stable and the
            # TRACK3_STAMP filter below preserves exact identity.
            query_terms.append(f"{stamp[:match.start()]}_w{match.group(1)}_")
    return list(dict.fromkeys(query_terms))


def list_runs(stamp: str, page_size: int) -> list[dict[str, Any]]:
    query_terms = stamp_query_terms(stamp)

    rows_by_id: dict[str, dict[str, Any]] = {}
    for query in query_terms:
        current = 1
        while True:
            try:
                obj = run_merlin(
                    ["job", "list-run"],
                    {"job_name": query, "pageSize": page_size, "current": current},
                    timeout=20,
                    retries=1,
                )
            except RuntimeError as exc:
                print(f"warning: list_runs failed for {stamp} query={query!r}: {exc}", flush=True)
                break
            page = obj.get("list", [])
            for item in page:
                env = item.get("meta", {}).get("job_def_version", {}).get("env", {})
                if str(env.get("TRACK3_STAMP", "")) == stamp:
                    rows_by_id[str(item.get("id", ""))] = item
            total = int(obj.get("total", len(page)))
            if current * page_size >= total or not page:
                break
            current += 1
        if rows_by_id:
            break
    return list(rows_by_id.values())


def get_run_item(job_run_id: str) -> dict[str, Any]:
    obj = run_merlin(["job", "get-run"], {"job_run_id": job_run_id})
    item = obj.get("job_run")
    if not isinstance(item, dict):
        raise RuntimeError(f"get-run returned no job_run for {job_run_id}")
    return item


def get_env(item: dict[str, Any]) -> dict[str, str]:
    env = item.get("meta", {}).get("job_def_version", {}).get("env", {})
    return {str(key): str(value) for key, value in env.items()}


def default_env_value(recipe: str, key: str) -> str:
    return RECIPE_DEFAULT_ENV_VALUES.get(recipe, {}).get(key, DEFAULT_ENV_VALUES[key])


def get_trial_id(item: dict[str, Any]) -> str:
    return str(
        item.get("latest_trial_id")
        or item.get("meta", {}).get("arnold_trial_id")
        or item.get("meta", {}).get("arnold_job_info", {}).get("trialId")
        or ""
    )


def batch_label(batch_size: str) -> str:
    labels = {
        str(128 * 1024): "128k",
        str(512 * 1024): "512k",
        str(1024 * 1024): "1m",
        str(2 * 1024 * 1024): "2m",
    }
    return labels.get(batch_size, batch_size)


def gpu_family(value: str) -> str:
    normalized = value.upper().replace("NVIDIA_", "")
    for family in ("A100", "H20", "H800", "GB200"):
        if family in normalized:
            return family
    return normalized


def parse_logs(text: str) -> dict[str, Any]:
    last_step = None
    total_steps = None
    val_points: list[tuple[int, float]] = []
    for match in STEP_RE.finditer(text):
        step = int(match.group("step"))
        total = int(match.group("total"))
        last_step = step if last_step is None else max(last_step, step)
        total_steps = total
        val_raw = match.group("val")
        if val_raw is not None:
            try:
                val_points.append((step, float(val_raw)))
            except ValueError:
                val_points.append((step, math.nan))

    lower = text.lower()
    failure_kind = ""
    if "cuda out of memory" in lower or "outofmemoryerror" in lower:
        failure_kind = "oom"
    elif re.search(
        r"distbackenderror|nccl(?:\s+error|systemerror|unhandled|internalerror|"
        r"remoteerror|invalid|timeout)|watchdog caught collective|"
        r"collective operation timeout|connection closed by peer|pthread_join",
        lower,
    ):
        failure_kind = "nccl_or_node"
    elif "traceback" in lower or "runtimeerror" in lower or "attributeerror" in lower:
        failure_kind = "python_error"
    elif re.search(r"val_loss:nan|loss[=: ]+nan|\\bnan loss\\b", lower):
        failure_kind = "nan_or_divergence"

    return {
        "last_step": last_step,
        "total_steps": total_steps,
        "last_val_step": val_points[-1][0] if val_points else None,
        "last_val_loss": val_points[-1][1] if val_points else None,
        "best_val_loss": min((v for _, v in val_points), default=None),
        "num_val_points": len(val_points),
        "failure_kind": failure_kind,
    }


def parse_runtime_metadata(text: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for field, pattern in RUNTIME_PATTERNS.items():
        values = list(dict.fromkeys(re.findall(pattern, text)))
        metadata[field] = "|".join(values)

    host_values: list[str] = []
    host_ipv6_values: list[str] = []
    for match in re.finditer(r"Instance start on host\s+([^\r\n]+)", text):
        tokens = match.group(1).split()
        if tokens and tokens[0] not in host_values:
            host_values.append(tokens[0])
        ipv6 = next(
            (
                token
                for token in tokens
                if ":" in token and not token.lower().startswith(("fdbd:", "fe80:"))
            ),
            "",
        )
        if ipv6 and ipv6 not in host_ipv6_values:
            host_ipv6_values.append(ipv6)
    metadata["runtime_host_primary"] = "|".join(host_values)
    metadata["runtime_host_ipv6"] = "|".join(host_ipv6_values)
    return metadata


def list_log_urls(job_run_id: str, trial_id: str) -> dict[str, str]:
    if not job_run_id or not trial_id:
        return {}
    try:
        obj = run_merlin(
            ["job", "list-trial-logs"],
            {"job_run_id": job_run_id, "trial_id": trial_id},
        )
    except RuntimeError:
        return {}
    urls: dict[str, str] = {}
    for entry in obj.get("log_list", []):
        typ = entry.get("type")
        url = entry.get("url")
        if typ and url:
            urls[str(typ)] = str(url)
    return urls


def fetch_url(url: str, timeout: int) -> str:
    try:
        request = urllib.request.Request(
            url,
            headers={
                "Accept-Encoding": "identity",
                "Range": f"bytes=-{LOG_TAIL_BYTES}",
            },
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read(LOG_TAIL_BYTES).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        if exc.code == 416:
            return ""
        return f"\nLOG_FETCH_HTTP_ERROR {exc.code}\n"
    except Exception as exc:
        return f"\nLOG_FETCH_ERROR {type(exc).__name__}: {exc}\n"


def fetch_merlin_text(job_run_id: str, trial_id: str, timeout: int) -> str:
    urls = list_log_urls(job_run_id, trial_id)
    stdout = fetch_url(urls.get("stdout", ""), timeout) if urls.get("stdout") else ""
    stderr = fetch_url(urls.get("stderr", ""), timeout) if urls.get("stderr") else ""
    return f"{stderr}\n{stdout}"


def hdfs_cat(path: str, timeout: int) -> str:
    try:
        return subprocess.check_output(
            [HDFS_CLI, "dfs", "-cat", path],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ""


def hdfs_ls(path: str, timeout: int) -> list[str]:
    try:
        out = subprocess.check_output(
            [HDFS_CLI, "dfs", "-ls", path],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []
    files: list[str] = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 8 and not parts[-1].endswith("/"):
            files.append(parts[-1])
    return files


def read_run_dir_logs(stamp: str, case_id: str, timeout: int, *, probe_hdfs: bool) -> tuple[str, str]:
    if not stamp or not case_id:
        return "", ""
    local_dir = LOCAL_RUN_BASE / stamp / case_id
    text_parts: list[str] = []
    env_text = ""
    env_path = local_dir / "env.txt"
    if env_path.exists():
        env_text = env_path.read_text(encoding="utf-8", errors="replace")
    logs_dir = local_dir / "logs"
    if logs_dir.exists():
        for path in sorted(logs_dir.glob("*.txt")):
            text_parts.append(path.read_text(encoding="utf-8", errors="replace"))

    if text_parts or env_text or not probe_hdfs:
        return "\n".join(text_parts), env_text

    hdfs_dir = f"{HDFS_RUN_BASE.rstrip('/')}/{stamp}/{case_id}"
    env_text = hdfs_cat(f"{hdfs_dir}/env.txt", timeout)
    for path in hdfs_ls(f"{hdfs_dir}/logs", timeout):
        if path.endswith(".txt"):
            text_parts.append(hdfs_cat(path, timeout))
    return "\n".join(text_parts), env_text


def read_run_dir_exit_status(
    stamp: str,
    case_id: str,
    timeout: int,
    *,
    probe_hdfs: bool,
) -> str:
    if not stamp or not case_id:
        return ""
    local_path = LOCAL_RUN_BASE / stamp / case_id / "exit_status.txt"
    if local_path.is_file():
        return local_path.read_text(encoding="utf-8", errors="replace")
    if not probe_hdfs:
        return ""
    hdfs_path = (
        f"{HDFS_RUN_BASE.rstrip('/')}/{stamp}/{case_id}/exit_status.txt"
    )
    return hdfs_cat(hdfs_path, timeout)


def decode_wrapper_cases(env: dict[str, str]) -> list[dict[str, str]]:
    if env.get("TRACK3_WRAPPER_CASES_ZLIB_B64"):
        raw = zlib.decompress(
            base64.b64decode(env["TRACK3_WRAPPER_CASES_ZLIB_B64"])
        )
    elif env.get("TRACK3_WRAPPER_CASES_B64"):
        raw = base64.b64decode(env["TRACK3_WRAPPER_CASES_B64"])
    else:
        return []
    cases = json.loads(raw)
    if not isinstance(cases, list):
        raise ValueError("Wrapper cases payload must be a list")
    normalized: list[dict[str, str]] = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each wrapper case must be an object")
        child = {str(key): str(value) for key, value in case.items()}
        if not child.get("TRACK3_CASE_ID"):
            raise ValueError("Wrapper child is missing TRACK3_CASE_ID")
        normalized.append(child)
    return normalized


def load_packed_manifests(paths: list[Path]) -> dict[str, list[dict[str, str]]]:
    manifests: dict[str, list[dict[str, str]]] = {}
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, list) or not document:
            raise ValueError(f"packed manifest must be a non-empty case list: {path}")
        cases = []
        for case in document:
            env = case.get("env") if isinstance(case, dict) else None
            if not isinstance(env, dict) or not env.get("TRACK3_CASE_ID"):
                raise ValueError(f"packed manifest case is missing env identity: {path}")
            cases.append({str(key): str(value) for key, value in env.items()})
        stamps = {case.get("TRACK3_STAMP", "") for case in cases}
        if len(stamps) != 1 or "" in stamps:
            raise ValueError(f"packed manifest must contain exactly one stamp: {path}")
        stamp = next(iter(stamps))
        if stamp in manifests:
            raise ValueError(f"duplicate packed manifest stamp: {stamp}")
        manifests[stamp] = cases
    return manifests


def inject_packed_manifest(item: dict[str, Any], cases: list[dict[str, str]]) -> None:
    """Expose an embedded packed schedule through the collector wrapper contract."""

    raw = json.dumps(cases, sort_keys=True, separators=(",", ":")).encode()
    encoded = base64.b64encode(zlib.compress(raw)).decode("ascii")
    meta = item.setdefault("meta", {})
    version = meta.setdefault("job_def_version", {})
    env = version.setdefault("env", {})
    env.update(
        {
            "TRACK3_STAMP": cases[0]["TRACK3_STAMP"],
            "TRACK3_CASE_ID": "packed_manifest",
            "TRACK3_COORD": "wrapper",
            "TRACK3_WRAPPER_CASES_ZLIB_B64": encoded,
        }
    )


def child_status(outer_status: str, exit_status: str, run_dir_present: bool) -> str:
    token = exit_status.strip().splitlines()[0] if exit_status.strip() else ""
    if token:
        try:
            return "DONE" if int(token) == 0 else "FAILED"
        except ValueError:
            return "FAILED"
    if run_dir_present:
        if outer_status in {"FAILED", "FAILED_TO_LAUNCH", "STOPPED"}:
            return outer_status
        if outer_status == "DONE":
            return "FAILED"
        return "RUNNING"
    if outer_status in TERMINAL_STATUSES:
        return "NOT_STARTED"
    return "PENDING"


def has_complete_child_evidence(row: dict[str, Any]) -> bool:
    expected_steps = numeric(row.get("train_steps"))
    last_step = numeric(row.get("last_step"))
    last_val_step = numeric(row.get("last_val_step"))
    last_val_loss = numeric(row.get("last_val_loss"))
    return bool(
        expected_steps is not None
        and last_step is not None
        and last_step >= expected_steps
        and last_val_step is not None
        and last_val_step >= expected_steps
        and last_val_loss is not None
        and math.isfinite(last_val_loss)
    )


def parse_env_text(text: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("TRACK3_"):
            env[key] = value.strip()
    return env


def collect_log_text(
    *,
    stamp: str,
    case_id: str,
    job_run_id: str,
    trial_id: str,
    fetch_logs: bool,
    probe_hdfs: bool,
    timeout: int,
) -> tuple[str, dict[str, str], str, str]:
    run_dir_text, run_dir_env = read_run_dir_logs(stamp, case_id, timeout, probe_hdfs=probe_hdfs)
    if not fetch_logs:
        source = "run_dir" if run_dir_text else "none"
        return run_dir_text, parse_env_text(run_dir_env), source, ""
    merlin_text = fetch_merlin_text(job_run_id, trial_id, timeout)
    if run_dir_text:
        source = "run_dir+merlin_meta" if merlin_text.strip() else "run_dir"
        return run_dir_text, parse_env_text(run_dir_env), source, merlin_text
    source = "merlin_logs" if merlin_text.strip() else "none"
    return merlin_text, parse_env_text(run_dir_env), source, merlin_text


def row_from_item(
    stamp: str,
    item: dict[str, Any],
    *,
    fetch_logs: bool,
    probe_hdfs: bool,
    timeout: int,
    env_override: dict[str, str] | None = None,
    status_override: str | None = None,
    run_dir_artifacts: tuple[str, str] | None = None,
    runtime_text_override: str | None = None,
    allow_merlin_loss_fallback: bool = True,
) -> dict[str, Any]:
    env = dict(env_override) if env_override is not None else get_env(item)
    arnold = (
        item.get("meta", {})
        .get("job_def_version", {})
        .get("resource", {})
        .get("arnold_config", {})
    )
    roles = arnold.get("roles", [])
    role = roles[0] if roles else {}
    gpuv = str(role.get("gpuv", ""))
    queue_name = str(role.get("queueName", "") or role.get("queue_name", ""))
    job_def = item.get("meta", {}).get("job_def_version", {})
    image_meta = job_def.get("image_meta", {})
    container_image = str(
        image_meta.get("image_url", "")
        or job_def.get("icm_image", "")
        or job_def.get("image", "")
    )
    code_package_id = env.get("TRACK3_CODE_PACKAGE_ID", "")
    if not code_package_id and env.get("HDFS_CODE_TGZ"):
        code_package_id = env["HDFS_CODE_TGZ"].rstrip("/").rsplit("/", 1)[-1]
    hardware_family = env.get("TRACK3_HARDWARE_FAMILY", "") or gpu_family(gpuv)
    job_run_id = str(item.get("id", ""))
    trial_id = get_trial_id(item)
    status = status_override if status_override is not None else str(item.get("status", ""))
    case_id = env.get("TRACK3_CASE_ID", "")
    if run_dir_artifacts is None:
        text, run_dir_env, log_source, runtime_text = collect_log_text(
            stamp=stamp,
            case_id=case_id,
            job_run_id=job_run_id,
            trial_id=trial_id,
            fetch_logs=fetch_logs,
            probe_hdfs=probe_hdfs and status in {"DONE", "FAILED", "STOPPED"},
            timeout=timeout,
        )
    else:
        text, run_dir_env_text = run_dir_artifacts
        run_dir_env = parse_env_text(run_dir_env_text)
        runtime_text = runtime_text_override or ""
        if text:
            log_source = "run_dir+merlin_meta" if runtime_text.strip() else "run_dir"
        elif run_dir_env_text:
            log_source = "run_dir_metadata"
        else:
            log_source = "none"
    env.update({key: value for key, value in run_dir_env.items() if value})
    parsed = parse_logs(text) if text else {
        "last_step": None,
        "total_steps": None,
        "last_val_step": None,
        "last_val_loss": None,
        "best_val_loss": None,
        "num_val_points": 0,
        "failure_kind": "",
    }
    if (
        fetch_logs
        and allow_merlin_loss_fallback
        and log_source in {"run_dir", "run_dir+merlin_meta"}
        and status in {"DONE", "FAILED", "STOPPED"}
        and not parsed["num_val_points"]
    ):
        fallback_text = runtime_text
        if not fallback_text.strip():
            fallback_text = fetch_merlin_text(job_run_id, trial_id, timeout)
            runtime_text = fallback_text
        fallback_parsed = parse_logs(fallback_text) if fallback_text.strip() else None
        if fallback_parsed and (fallback_parsed["num_val_points"] or fallback_parsed["failure_kind"]):
            parsed = fallback_parsed
            log_source = "merlin_logs_fallback"
    if parsed["failure_kind"] and status != "FAILED":
        parsed["failure_kind"] = ""
    runtime_metadata = parse_runtime_metadata(runtime_text or text)

    row: dict[str, Any] = {
        "stamp": stamp,
        "status": status,
        "failure_kind": parsed["failure_kind"],
        "recipe": env.get("TRACK3_RECIPE", ""),
        "batch": batch_label(env.get("TRACK3_BATCH_SIZE", "")),
        "batch_size": env.get("TRACK3_BATCH_SIZE", ""),
        "coord": env.get("TRACK3_COORD", ""),
        "case_id": case_id,
        "train_steps": env.get("TRACK3_TRAIN_STEPS", ""),
        "base_lr_scale": env.get("TRACK3_BASE_LR_SCALE", ""),
        "hidden_cooldown_frac": env.get("TRACK3_H_COOLDOWN_FRAC", ""),
        "hardware_family": hardware_family,
        "cluster_name": str(arnold.get("clusterName", "")),
        "queue_name": queue_name,
        "gpuv": gpuv,
        "container_image": container_image,
        **runtime_metadata,
        "code_package_id": code_package_id,
        "seed": env.get("TRACK3_SEED", env.get("KL_SOAP_SEED", "")),
        "data_id": env.get("TRACK3_DATA_HDFS", ""),
        "job_run_id": job_run_id,
        "trial_id": trial_id,
        "job_name": item.get("job_def_name", ""),
        "log_source": log_source,
        **parsed,
    }
    recipe = row["recipe"]
    for key in ENV_KEYS:
        row[key.removeprefix("TRACK3_").lower()] = env.get(key) or default_env_value(recipe, key)
    row["coord_value"] = coord_value(row)
    return row


def rows_from_item(
    stamp: str,
    item: dict[str, Any],
    *,
    fetch_logs: bool,
    probe_hdfs: bool,
    timeout: int,
) -> list[dict[str, Any]]:
    outer_env = get_env(item)
    if outer_env.get("TRACK3_COORD") != "wrapper":
        return [
            row_from_item(
                stamp,
                item,
                fetch_logs=fetch_logs,
                probe_hdfs=probe_hdfs,
                timeout=timeout,
            )
        ]

    cases = decode_wrapper_cases(outer_env)
    if not cases:
        raise ValueError(
            f"Wrapper {item.get('id', '')} has no decodable child cases"
        )
    job_run_id = str(item.get("id", ""))
    trial_id = get_trial_id(item)
    outer_status = str(item.get("status", ""))
    runtime_text = (
        fetch_merlin_text(job_run_id, trial_id, timeout) if fetch_logs else ""
    )
    rows: list[dict[str, Any]] = []
    for child_env in cases:
        case_id = child_env["TRACK3_CASE_ID"]
        child_stamp = child_env.get("TRACK3_STAMP", stamp)
        run_text, run_env_text = read_run_dir_logs(
            child_stamp,
            case_id,
            timeout,
            probe_hdfs=probe_hdfs,
        )
        exit_status = read_run_dir_exit_status(
            child_stamp,
            case_id,
            timeout,
            probe_hdfs=probe_hdfs,
        )
        status = child_status(
            outer_status,
            exit_status,
            bool(run_text or run_env_text or exit_status.strip()),
        )
        row = row_from_item(
            child_stamp,
            item,
            fetch_logs=fetch_logs,
            probe_hdfs=probe_hdfs,
            timeout=timeout,
            env_override=child_env,
            status_override=status,
            run_dir_artifacts=(run_text, run_env_text),
            runtime_text_override=runtime_text,
            allow_merlin_loss_fallback=False,
        )
        if status == "DONE" and not has_complete_child_evidence(row):
            row["status"] = "FAILED"
            row["failure_kind"] = "incomplete_artifact"
        row["wrapper_case_id"] = outer_env.get("TRACK3_CASE_ID", "")
        row["job_name"] = f"{item.get('job_def_name', '')}:{case_id}"
        rows.append(row)
    return rows


def numeric(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def coord_value(row: dict[str, Any]) -> str:
    coord = str(row.get("coord", ""))
    if coord == "center":
        return "center"
    column = COORD_VALUE_COLUMNS.get(coord)
    if not column:
        return ""
    return str(row.get(column, ""))


def same_value(left: float, right: float) -> bool:
    # Older accepted recipes serialize some sqrt(2)-scaled centers to six digits.
    return math.isclose(left, right, rel_tol=1e-5, abs_tol=1e-12)


def expected_values_for_coord(coord: str, center_value: float, grid_mode: str) -> list[float]:
    return cd_policy.coordinate_values(
        coord,
        center_value,
        mode=grid_mode,
        wd_grid=standard_cd.STANDARD_WD_CD_GRID,
    )


def expected_arm_count(coord: str, center_value: float, grid_mode: str) -> int:
    return len(expected_values_for_coord(coord, center_value, grid_mode))


def best_attempt(rows: list[dict[str, Any]]) -> dict[str, Any]:
    done = [row for row in rows if row.get("status") == "DONE"]
    if done:
        return min(done, key=lambda row: numeric(row.get("last_val_loss")) or float("inf"))
    running = [row for row in rows if row.get("status") == "RUNNING"]
    if running:
        return max(running, key=lambda row: numeric(row.get("last_step")) or -1)
    return max(rows, key=lambda row: numeric(row.get("last_step")) or -1)


def terminal_losses(rows: list[dict[str, Any]]) -> list[float]:
    return [
        float(loss)
        for row in rows
        if row.get("status") == "DONE"
        and (loss := numeric(row.get("last_val_loss"))) is not None
    ]


def print_summary(
    rows: list[dict[str, Any]],
    threshold: float,
    close_gain_threshold: float,
    close_gain_required_repeats: int,
    grid_mode: str,
) -> None:
    print("rows", len(rows))
    print("status", dict(Counter(row["status"] for row in rows)))
    print("log_source", dict(Counter(row["log_source"] for row in rows)))
    failures = Counter(row["failure_kind"] for row in rows if row["failure_kind"])
    if failures:
        print("failure_kind", dict(failures))

    grouped: dict[
        tuple[str, str, str, str, str, str, str, str, str, str, str, str, str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["recipe"],
                row["batch"],
                row.get("hardware_family", ""),
                row.get("cluster_name", ""),
                row.get("queue_name", ""),
                row.get("gpuv", ""),
                row.get("container_image", ""),
                row.get("runtime_device_type", ""),
                row.get("runtime_node_spec", ""),
                row.get("runtime_rdma_bigpod", ""),
                row.get("runtime_cuda_driver", ""),
                row.get("runtime_cuda_toolkit", ""),
                row.get("code_package_id", ""),
                row.get("seed", ""),
                row.get("data_id", ""),
            )
        ].append(row)

    print("settings")
    for key in sorted(grouped):
        group = grouped[key]
        (
            recipe,
            batch,
            hardware,
            cluster,
            queue_name,
            gpuv,
            container_image,
            runtime_device_type,
            runtime_node_spec,
            runtime_rdma_bigpod,
            runtime_cuda_driver,
            runtime_cuda_toolkit,
            code_package,
            seed,
            data_id,
        ) = key
        cohort_complete = bool(
            hardware
            and cluster
            and queue_name
            and gpuv
            and container_image
            and runtime_device_type
            and runtime_node_spec
            and runtime_rdma_bigpod
            and runtime_cuda_driver
            and runtime_cuda_toolkit
            and code_package
            and seed
            and data_id
        )
        counts = Counter(row["status"] for row in group)
        center_attempts = [row for row in group if row["coord"] == "center"]
        center = best_attempt(center_attempts) if center_attempts else None
        center_losses = terminal_losses(center_attempts)
        center_loss = statistics.median(center_losses) if center_losses else None
        non_center_coords = sorted({str(row["coord"]) for row in group if row["coord"] != "center"})
        candidates: list[tuple[float, int, dict[str, Any], list[float]]] = []
        coord_ready: list[bool] = []
        for coord in non_center_coords:
            attempts_by_value: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in group:
                if row["coord"] != coord or not row.get("coord_value"):
                    continue
                value = numeric(row["coord_value"])
                value_key = f"{value:.12g}" if value is not None else str(row["coord_value"])
                attempts_by_value[value_key].append(row)
            for attempts in attempts_by_value.values():
                losses = terminal_losses(attempts)
                if losses:
                    candidates.append(
                        (statistics.median(losses), len(losses), best_attempt(attempts), losses)
                    )

            center_value = numeric(center.get(COORD_VALUE_COLUMNS.get(coord, "")) if center else None)
            required_keys: list[str] = []
            if center_value is not None:
                for expected_value in expected_values_for_coord(coord, center_value, grid_mode):
                    matching_key = next(
                        (
                            value
                            for value in attempts_by_value
                            if numeric(value) is not None and same_value(float(value), expected_value)
                        ),
                        None,
                    )
                    if matching_key is not None:
                        required_keys.append(matching_key)
            required_attempts = [best_attempt(attempts_by_value[value]) for value in required_keys]
            expected = expected_arm_count(coord, center_value, grid_mode) if center_value is not None else 0
            coord_ready.append(
                len(required_keys) >= expected
                and bool(required_attempts)
                and all(row.get("status") == "DONE" for row in required_attempts)
            )
        candidate_metrics = list(candidates)
        if center_loss is not None and center is not None:
            candidate_metrics.append((center_loss, len(center_losses), center, center_losses))
        best_metric = min(candidate_metrics, key=lambda metric: metric[0]) if candidate_metrics else None
        best = best_metric[2] if best_metric else None
        best_loss = best_metric[0] if best_metric else None
        best_matching_runs = best_metric[1] if best_metric else 0
        best_losses = best_metric[3] if best_metric else []
        gain = None
        if center_loss is not None and best_loss is not None:
            gain = center_loss - best_loss
        ready = (
            center is not None
            and center.get("status") == "DONE"
            and bool(coord_ready)
            and all(coord_ready)
        )
        screening_positive = gain is not None and gain >= threshold
        required_repeats = (
            close_gain_required_repeats
            if screening_positive and gain is not None and gain < close_gain_threshold
            else 2
        )
        confirmation_ready = (
            len(center_losses) >= required_repeats
            and best_matching_runs >= required_repeats
        )
        direction_consistent = bool(
            best is not None
            and center is not None
            and best is not center
            and confirmation_ready
            and max(best_losses) < min(center_losses)
        )
        accepted = (
            ready
            and cohort_complete
            and confirmation_ready
            and screening_positive
            and direction_consistent
        )
        cohort = (
            f"{hardware}/{cluster}/{queue_name}/{gpuv}/{container_image}/"
            f"device={runtime_device_type}/node={runtime_node_spec}/bigpod={runtime_rdma_bigpod}/"
            f"cuda={runtime_cuda_driver}:{runtime_cuda_toolkit}/{code_package}/seed={seed}/data={data_id}"
        )
        print(
            f"  {(recipe, batch)} cohort={cohort}: status={dict(counts)} grid_mode={grid_mode} "
            f"scope=collected_coordinates ready={ready} "
            f"cohort_complete={cohort_complete} coords_ready={sum(coord_ready)}/{len(coord_ready)} "
            f"center_median={center_loss} center_n={len(center_losses)} "
            f"best={best.get('stamp') if best else ''}/{best.get('case_id') if best else ''} "
            f"best_median={best_loss} best_n={best_matching_runs} "
            f"gain={gain} screening_positive={screening_positive} "
            f"required_repeats={required_repeats} confirmation_ready={confirmation_ready} "
            f"direction_consistent={direction_consistent} "
            f"accept={accepted}"
        )


def write_outputs(rows: list[dict[str, Any]], output_dir: Path, tag: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{tag}.json"
    csv_path = output_dir / f"{tag}.csv"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    fields = [
        "stamp",
        "status",
        "failure_kind",
        "recipe",
        "batch",
        "coord",
        "coord_value",
        "case_id",
        "wrapper_case_id",
        "last_step",
        "total_steps",
        "last_val_step",
        "last_val_loss",
        "best_val_loss",
        "num_val_points",
        "matrix_lr_mult",
        "precond_lr_mult",
        "matrix_beta1_om_mult",
        "matrix_beta2_om_mult",
        "shampoo_beta_om_mult",
        "aux_lr_mult",
        "aux_beta1_om_mult",
        "aux_beta2_om_mult",
        "aux_cooldown_frac",
        "matrix_wd_peak",
        "aux_wd_peak",
        "wd_warmup_frac",
        "base_lr_scale",
        "hidden_cooldown_frac",
        "hardware_family",
        "cluster_name",
        "queue_name",
        "gpuv",
        "container_image",
        "runtime_device_type",
        "runtime_node_spec",
        "runtime_node_vendor",
        "runtime_rdma_bigpod",
        "runtime_rdma_minipod",
        "runtime_rdma_switch",
        "runtime_cuda_driver",
        "runtime_cuda_toolkit",
        "runtime_host_primary",
        "runtime_host_ipv6",
        "code_package_id",
        "seed",
        "data_id",
        "batch_size",
        "train_steps",
        "job_run_id",
        "trial_id",
        "job_name",
        "log_source",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


def load_terminal_cache(
    path: Path | None,
) -> dict[tuple[str, str], dict[str, str]]:
    if path is None or not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    cache: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if not row.get("job_run_id") or row.get("status") not in TERMINAL_STATUSES:
            continue
        if row.get("status") == "DONE" and not has_complete_child_evidence(row):
            continue
        cache[(row["job_run_id"], row.get("case_id", ""))] = row
    return cache


def cached_rows_for_item(
    item: dict[str, Any],
    terminal_cache: dict[tuple[str, str], dict[str, str]],
) -> list[dict[str, str]] | None:
    outer_status = str(item.get("status", ""))
    if outer_status not in TERMINAL_STATUSES:
        return None
    env = get_env(item)
    if env.get("TRACK3_COORD") == "wrapper":
        case_ids = [case["TRACK3_CASE_ID"] for case in decode_wrapper_cases(env)]
    else:
        case_ids = [env.get("TRACK3_CASE_ID", "")]
    if not case_ids or any(not case_id for case_id in case_ids):
        return None
    job_run_id = str(item.get("id", ""))
    cached = [terminal_cache.get((job_run_id, case_id)) for case_id in case_ids]
    if any(row is None for row in cached):
        return None
    rows = [dict(row) for row in cached if row is not None]
    if env.get("TRACK3_COORD") != "wrapper" and rows[0].get("status") != outer_status:
        return None
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stamps", nargs="+", required=True)
    parser.add_argument(
        "--job-run-ids",
        nargs="*",
        default=None,
        help=(
            "Collect only these job run IDs. IDs not found by stamp search are "
            "resolved directly with get-run."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--tag", default=time.strftime("core_cd_collect_%Y%m%d_%H%M%S"))
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument(
        "--list-run-workers",
        type=int,
        default=8,
        help="Concurrent read-only Merlin list-run queries.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--close-gain-threshold",
        type=float,
        default=None,
        help="Require extra repeats when a positive gain is smaller than this value.",
    )
    parser.add_argument(
        "--noise-calibration",
        type=Path,
        default=OUTPUT_DIR / "cd_noise_calibration_latest.json",
    )
    parser.add_argument(
        "--close-gain-required-repeats",
        type=int,
        default=cd_policy.DEFAULT_CLOSE_GAIN_REQUIRED_REPEATS,
    )
    parser.add_argument("--grid-mode", choices=("local", "full"), default="local")
    parser.add_argument("--no-fetch-logs", action="store_true")
    parser.add_argument("--no-probe-hdfs", action="store_true", help="Skip HDFS run_dir probes for terminal jobs.")
    parser.add_argument("--statuses", nargs="*", default=None, help="Only fetch Merlin logs for these statuses.")
    parser.add_argument("--max-log-jobs", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=8)
    parser.add_argument(
        "--reuse-terminal-from",
        type=Path,
        default=None,
        help="Reuse parsed rows for immutable terminal jobs from a prior collect CSV.",
    )
    parser.add_argument(
        "--packed-manifests",
        type=Path,
        nargs="*",
        default=[],
        help="Local case manifests for packed jobs whose schedule is embedded in entrypoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_run_workers < 1:
        raise SystemExit("--list-run-workers must be positive")
    try:
        noise_policy = cd_policy.load_noise_policy(args.noise_calibration)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    threshold = args.threshold if args.threshold is not None else noise_policy.acceptance_threshold
    close_gain_threshold = (
        args.close_gain_threshold
        if args.close_gain_threshold is not None
        else noise_policy.close_gain_threshold
    )
    if close_gain_threshold < threshold:
        raise SystemExit("--close-gain-threshold must be >= --threshold")
    if args.close_gain_required_repeats < 2:
        raise SystemExit("--close-gain-required-repeats must be >= 2")
    socket.setdefaulttimeout(args.timeout)
    print(
        f"noise_policy={noise_policy.status} threshold={threshold:g} "
        f"close_gain_threshold={close_gain_threshold:g} source={noise_policy.source}",
        flush=True,
    )
    statuses = set(args.statuses) if args.statuses else None
    terminal_cache = load_terminal_cache(args.reuse_terminal_from)
    try:
        packed_manifests = load_packed_manifests(args.packed_manifests)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(f"terminal_cache={len(terminal_cache)}", flush=True)
    fetched = 0
    rows: list[dict[str, Any]] = []
    requested_job_ids = set(args.job_run_ids or [])
    with ThreadPoolExecutor(max_workers=args.list_run_workers) as pool:
        stamp_items = list(
            zip(
                args.stamps,
                pool.map(lambda stamp: list_runs(stamp, args.page_size), args.stamps),
            )
        )
    if requested_job_ids:
        resolved_job_ids = {
            str(item.get("id", ""))
            for _, items in stamp_items
            for item in items
        }
        missing_job_ids = sorted(requested_job_ids - resolved_job_ids)
        if missing_job_ids:
            with ThreadPoolExecutor(max_workers=args.list_run_workers) as pool:
                direct_items = list(pool.map(get_run_item, missing_job_ids))
            items_by_stamp = {stamp: list(items) for stamp, items in stamp_items}
            requested_stamps = set(args.stamps)
            for item in direct_items:
                env = get_env(item)
                stamp = env.get("TRACK3_STAMP", "")
                if not stamp and len(args.stamps) == 1 and args.stamps[0] in packed_manifests:
                    inject_packed_manifest(item, packed_manifests[args.stamps[0]])
                    env = get_env(item)
                    stamp = env.get("TRACK3_STAMP", "")
                if stamp not in requested_stamps:
                    raise SystemExit(
                        f"job {item.get('id', '')} has TRACK3_STAMP={stamp!r}, "
                        "which was not requested"
                    )
                items_by_stamp[stamp].append(item)
            stamp_items = [(stamp, items_by_stamp[stamp]) for stamp in args.stamps]
    for stamp, items in stamp_items:
        if requested_job_ids:
            items = [item for item in items if str(item.get("id", "")) in requested_job_ids]
        print(f"{stamp}: {len(items)} runs", flush=True)
        for index, item in enumerate(items, start=1):
            if stamp in packed_manifests and not get_env(item).get("TRACK3_STAMP"):
                inject_packed_manifest(item, packed_manifests[stamp])
            status = str(item.get("status", ""))
            job_run_id = str(item.get("id", ""))
            cached = cached_rows_for_item(item, terminal_cache)
            if cached is not None:
                print(f"reuse: {index}/{len(items)} {status} {item.get('job_def_name', '')}", flush=True)
                rows.extend(cached)
                continue
            fetch_logs = not args.no_fetch_logs
            if statuses is not None and status not in statuses:
                fetch_logs = False
            if args.max_log_jobs is not None and fetched >= args.max_log_jobs:
                fetch_logs = False
            if fetch_logs:
                fetched += 1
                print(f"fetch {fetched}: {index}/{len(items)} {status} {item.get('job_def_name', '')}", flush=True)
            rows.extend(
                rows_from_item(
                    stamp,
                    item,
                    fetch_logs=fetch_logs,
                    probe_hdfs=not args.no_probe_hdfs,
                    timeout=args.timeout,
                )
            )

    rows.sort(key=lambda row: (row["stamp"], row["recipe"], row["batch"], row["case_id"]))
    print_summary(
        rows,
        threshold,
        close_gain_threshold,
        args.close_gain_required_repeats,
        args.grid_mode,
    )
    write_outputs(rows, args.output_dir, args.tag)


if __name__ == "__main__":
    main()
