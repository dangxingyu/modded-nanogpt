#!/usr/bin/env python3
"""Run one LPT-assigned queue for the final-recipe ablation campaign."""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import queue
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path


ENTRYPOINTS = {
    "hyperball": Path(
        "records/track_3_optimization/batch_size_cd/merlin_core_entrypoint.sh"
    ),
    "standard_wd": Path(
        "records/track_3_optimization/batch_size_cd/merlin_standard_wd_entrypoint.sh"
    ),
}

STEP_RE = re.compile(r"^step:(?P<step>\d+)/(?P<total>\d+)")
VAL_STEP_RE = re.compile(
    r"^step:(?P<step>\d+)/(?P<total>\d+)\s+val_loss:"
)
# Keep this fail-closed skip contract aligned with
# final_recipe_ablation_report.RUNTIME_ENV_KEYS. The report module is not part
# of the minimal worker archive, so the runtime cannot import it directly.
RUNTIME_ENV_KEYS = (
    "TRACK3_CASE_ID",
    "TRACK3_RECIPE",
    "TRACK3_BATCH_SIZE",
    "TRACK3_BATCH_RATIO",
    "TRACK3_TRAIN_STEPS",
    "TRACK3_COORD",
    "TRACK3_MATRIX_LR_MULT",
    "TRACK3_AUX_LR_MULT",
    "TRACK3_MATRIX_BETA1_OM_MULT",
    "TRACK3_MATRIX_BETA2_OM_MULT",
    "TRACK3_PRECOND_LR_MULT",
    "TRACK3_SHAMPOO_BETA_OM_MULT",
    "TRACK3_AUX_BETA1_OM_MULT",
    "TRACK3_AUX_BETA2_OM_MULT",
    "TRACK3_AUX_COOLDOWN_FRAC",
    "TRACK3_MATRIX_WD_PEAK",
    "TRACK3_AUX_WD_PEAK",
    "TRACK3_WD_WARMUP_FRAC",
    "TRACK3_DIST_TIMEOUT_MINUTES",
    "TRACK3_BASE_LR_SCALE",
    "TRACK3_H_COOLDOWN_FRAC",
)
# The first compiled step for Shampoo/KL-SOAP can legitimately spend more than
# five minutes in graph/code generation on a cold H20 worker.  This timeout is
# infrastructure-only: it does not alter the training graph, schedule, or
# terminal-loss contract.
DEFAULT_STARTUP_TIMEOUT_SECONDS = 10 * 60
DEFAULT_STALL_TIMEOUT_SECONDS = 5 * 60
DEFAULT_POST_TERMINAL_GRACE_SECONDS = 3 * 60
DEFAULT_COMPLETION_PREFLIGHT_TIMEOUT_SECONDS = 2 * 60
DEFAULT_COMPLETION_PREFLIGHT_ATTEMPTS = 5
DEFAULT_COMPLETION_PREFLIGHT_RETRY_SECONDS = 2
DEFAULT_TERMINATE_GRACE_SECONDS = 30
DEFAULT_CLEANUP_GRACE_SECONDS = 2
DEFAULT_MAX_ATTEMPTS_PER_WORKER = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    return parser.parse_args()


def run_dir(env: dict[str, str]) -> Path:
    return (
        Path(env["TRACK3_OUTPUT_BASE"])
        / env["TRACK3_STAMP"]
        / env["TRACK3_CASE_ID"]
    )


def persisted_env_matches(env: dict[str, str]) -> bool:
    try:
        observed: dict[str, str] = {}
        for line in (run_dir(env) / "env.txt").read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                observed[key] = value
        categorical = {"TRACK3_CASE_ID", "TRACK3_RECIPE", "TRACK3_COORD"}
        for key in RUNTIME_ENV_KEYS:
            expected = str(env.get(key, ""))
            actual = observed.get(key)
            if actual is None:
                return False
            if key in categorical:
                if actual != expected:
                    return False
            else:
                try:
                    if not math.isclose(
                        float(actual),
                        float(expected),
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    ):
                        return False
                except ValueError:
                    return False
        return True
    except OSError:
        return False


def completed_case(env: dict[str, str]) -> bool:
    try:
        if (run_dir(env) / "exit_status.txt").read_text(
            encoding="utf-8"
        ).strip() != "0":
            return False
        if not persisted_env_matches(env):
            return False
        expected_step = int(env["TRACK3_TRAIN_STEPS"])
        if expected_step < 1:
            return False
        log_paths = sorted((run_dir(env) / "logs").glob("*.txt"))
        for log_path in log_paths:
            with log_path.open(encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    match = VAL_STEP_RE.match(line)
                    if match and (
                        int(match.group("step")) == expected_step
                        and int(match.group("total")) == expected_step
                    ):
                        return True
        return False
    except (FileNotFoundError, KeyError, OSError, ValueError):
        return False


class CompletionPreflightTimeout(TimeoutError):
    pass


def completed_case_preflight(
    env: dict[str, str], timeout_seconds: float | None = None,
) -> tuple[bool, bool]:
    """Bound remote completion lookup before scientific work begins.

    The completion directory is on an HDFS mount. During NNProxy overload a
    metadata read can otherwise block before ``CASE_START`` indefinitely, so
    the normal training watchdog never gets a chance to act. A timeout fails
    closed (the case is not declared complete) and lets the scheduled run
    proceed; callers also skip remote evidence archival after a timed-out
    lookup to avoid immediately repeating the same blocked metadata request.
    """

    timeout = (
        timeout_from_env(
            env,
            "TRACK3_COMPLETION_PREFLIGHT_TIMEOUT_SECONDS",
            DEFAULT_COMPLETION_PREFLIGHT_TIMEOUT_SECONDS,
        )
        if timeout_seconds is None else float(timeout_seconds)
    )
    if timeout <= 0:
        return completed_case(env), False
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("completed_case_preflight requires the main thread")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0)

    def alarm_handler(_signum: int, _frame: object) -> None:
        raise CompletionPreflightTimeout

    signal.signal(signal.SIGALRM, alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        attempts = positive_int_from_env(
            env,
            "TRACK3_COMPLETION_PREFLIGHT_ATTEMPTS",
            DEFAULT_COMPLETION_PREFLIGHT_ATTEMPTS,
        )
        retry_seconds = timeout_from_env(
            env,
            "TRACK3_COMPLETION_PREFLIGHT_RETRY_SECONDS",
            DEFAULT_COMPLETION_PREFLIGHT_RETRY_SECONDS,
        )
        for attempt in range(1, attempts + 1):
            if completed_case(env):
                if attempt > 1:
                    print(
                        "TRACK3_FINAL_ABLATION_COMPLETION_PREFLIGHT_RECOVERED "
                        f"case_id={env.get('TRACK3_CASE_ID', 'unknown')} "
                        f"attempt={attempt}/{attempts}",
                        flush=True,
                    )
                return True, False
            if attempt < attempts and retry_seconds > 0:
                time.sleep(retry_seconds)
        return False, False
    except CompletionPreflightTimeout:
        print(
            "TRACK3_FINAL_ABLATION_COMPLETION_PREFLIGHT_TIMEOUT "
            f"case_id={env.get('TRACK3_CASE_ID', 'unknown')} "
            f"timeout_seconds={timeout:g}",
            flush=True,
        )
        return False, True
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def terminal_validation_in_logs(log_dir: Path, expected_step: int) -> bool:
    """Require an exact full-horizon validation record in a local log."""

    if expected_step < 1:
        return False
    try:
        for log_path in sorted(log_dir.glob("*.txt")):
            with log_path.open(encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    match = VAL_STEP_RE.match(line)
                    if match and (
                        int(match.group("step")) == expected_step
                        and int(match.group("total")) == expected_step
                    ):
                        return True
    except OSError:
        return False
    return False


def promote_post_terminal_completion(env: dict[str, str]) -> bool:
    """Persist a completed run whose launcher hung after terminal validation.

    This is deliberately fail-closed: stdout alone is insufficient.  The
    local training log must contain the exact expected terminal validation
    step before the worker may reconstruct the normal entrypoint EXIT
    contract.  This recovers only launcher teardown hangs after all scientific
    computation has completed.
    """

    try:
        expected_step = int(env["TRACK3_TRAIN_STEPS"])
    except (KeyError, ValueError):
        return False
    local_logs = Path("logs")
    if not terminal_validation_in_logs(local_logs, expected_step):
        return False
    case_dir = run_dir(env)
    try:
        case_dir.mkdir(parents=True, exist_ok=True)
        persisted_logs = case_dir / "logs"
        if persisted_logs.exists():
            shutil.rmtree(persisted_logs)
        shutil.copytree(local_logs, persisted_logs)
        (case_dir / "env.txt").write_text(
            "".join(f"{key}={env.get(key, '')}\n" for key in RUNTIME_ENV_KEYS),
            encoding="utf-8",
        )
        (case_dir / "exit_status.txt").write_text("0\n", encoding="utf-8")
    except OSError:
        return False
    return completed_case(env)


def archive_incomplete_evidence(
    env: dict[str, str], attempt: int
) -> Path | None:
    """Preserve stale/failed evidence before a clean retry of the same case."""

    case_dir = run_dir(env)
    members = ("logs", "env.txt", "exit_status.txt", "train_materialized.py")
    existing = [case_dir / member for member in members if (case_dir / member).exists()]
    if not existing:
        return None
    history_dir = (
        case_dir
        / "attempt_history"
        / f"attempt_{attempt:02d}_preexisting_{time.time_ns()}"
    )
    history_dir.mkdir(parents=True, exist_ok=False)
    for source in existing:
        shutil.move(str(source), str(history_dir / source.name))
    return history_dir


def local_data_ready() -> bool:
    data_dir = Path("data/fineweb10B")
    return (data_dir / "fineweb_val_000000.bin").is_file() and any(
        data_dir.glob("fineweb_train_*.bin")
    )


def timeout_from_env(env: dict[str, str], name: str, default: float) -> float:
    value = float(env.get(name, default))
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def positive_int_from_env(
    env: dict[str, str], name: str, default: int
) -> int:
    value = int(env.get(name, default))
    if value < 1:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def terminate_process_group(process: subprocess.Popen[str], grace_seconds: float) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def cleanup_process_group(pgid: int, grace_seconds: float) -> None:
    """Remove compiler/worker descendants left after the case leader exits."""

    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_with_progress_watchdog(
    command: list[str], env: dict[str, str]
) -> tuple[int, str | None]:
    """Tee one case and recover a process group that stops emitting steps."""

    startup_timeout = timeout_from_env(
        env,
        "TRACK3_CASE_STARTUP_TIMEOUT_SECONDS",
        DEFAULT_STARTUP_TIMEOUT_SECONDS,
    )
    stall_timeout = timeout_from_env(
        env,
        "TRACK3_CASE_STALL_TIMEOUT_SECONDS",
        DEFAULT_STALL_TIMEOUT_SECONDS,
    )
    post_terminal_grace = timeout_from_env(
        env,
        "TRACK3_CASE_POST_TERMINAL_GRACE_SECONDS",
        DEFAULT_POST_TERMINAL_GRACE_SECONDS,
    )
    terminate_grace = timeout_from_env(
        env,
        "TRACK3_CASE_TERMINATE_GRACE_SECONDS",
        DEFAULT_TERMINATE_GRACE_SECONDS,
    )
    cleanup_grace = timeout_from_env(
        env,
        "TRACK3_CASE_CLEANUP_GRACE_SECONDS",
        DEFAULT_CLEANUP_GRACE_SECONDS,
    )
    # The training entrypoints are Python programs whose stdout is piped through
    # this worker.  Without unbuffered output, long 128K runs can retain every
    # ``step:`` heartbeat until process exit, causing the startup watchdog to
    # mistake an almost-complete healthy run for a no-progress hang.  This only
    # changes log transport; it does not change the training graph or values.
    child_env = dict(env)
    child_env.setdefault("PYTHONUNBUFFERED", "1")
    process = subprocess.Popen(
        command,
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    events: queue.Queue[tuple[str, float]] = queue.Queue()

    expected_step = int(env.get("TRACK3_TRAIN_STEPS", "0"))

    def tee(stream: object, sink: object, inspect_steps: bool) -> None:
        for line in stream:  # type: ignore[union-attr]
            print(line, end="", file=sink, flush=True)
            if inspect_steps:
                observed_at = time.monotonic()
                if STEP_RE.match(line):
                    events.put(("step", observed_at))
                terminal = VAL_STEP_RE.match(line)
                if terminal and (
                    int(terminal.group("step")) == expected_step
                    and int(terminal.group("total")) == expected_step
                ):
                    events.put(("terminal", observed_at))

    threads = [
        threading.Thread(
            target=tee,
            args=(process.stdout, sys.stdout, True),
            daemon=True,
        ),
        threading.Thread(
            target=tee,
            args=(process.stderr, sys.stderr, False),
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()

    started = time.monotonic()
    last_step_at: float | None = None
    terminal_at: float | None = None
    watchdog_reason: str | None = None
    while process.poll() is None:
        try:
            event, observed_at = events.get(timeout=1.0)
            if event == "step":
                last_step_at = observed_at
            elif event == "terminal":
                terminal_at = observed_at
        except queue.Empty:
            pass
        now = time.monotonic()
        if (
            terminal_at is not None
            and post_terminal_grace > 0
            and now - terminal_at > post_terminal_grace
        ):
            watchdog_reason = "post_terminal_exit_timeout"
        elif (
            last_step_at is None
            and startup_timeout > 0
            and now - started > startup_timeout
        ):
            watchdog_reason = "startup_no_step_timeout"
        elif (
            last_step_at is not None
            and stall_timeout > 0
            and now - last_step_at > stall_timeout
        ):
            watchdog_reason = "step_heartbeat_timeout"
        if watchdog_reason is not None:
            print(
                "TRACK3_FINAL_ABLATION_CASE_WATCHDOG "
                f"reason={watchdog_reason} pid={process.pid} "
                f"startup_timeout_seconds={startup_timeout:g} "
                f"stall_timeout_seconds={stall_timeout:g}",
                flush=True,
            )
            terminate_process_group(process, terminate_grace)
            break

    return_code = process.wait()
    # torch.compile may leave forked Inductor workers alive after the shell and
    # rank processes exit.  The per-case session makes it safe to clean this
    # process group without touching the packed worker or the next case.
    cleanup_process_group(process.pid, cleanup_grace)
    for thread in threads:
        thread.join(timeout=5)
    if (
        watchdog_reason == "post_terminal_exit_timeout"
        and promote_post_terminal_completion(env)
    ):
        return 0, "post_terminal_exit_timeout_recovered"
    return return_code, watchdog_reason


def write_worker_summary(
    schedule: dict[str, object],
    worker_index: int,
    statuses: list[dict[str, object]],
) -> Path:
    output_base = Path(str(schedule["output_base"]))
    stamp = str(schedule["stamp"])
    summary_dir = output_base / stamp / "_packed_workers"
    summary_dir.mkdir(parents=True, exist_ok=True)
    path = summary_dir / f"worker_{worker_index:02d}.json"
    payload = {
        "schema": "track3_final_recipe_ablation_worker_v1",
        "stamp": stamp,
        "batch": schedule["batch"],
        "worker_index": worker_index,
        "statuses": statuses,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def wait_for_pack_drain(
    schedule: dict[str, object], worker_index: int, timeout_seconds: float
) -> bool:
    """Keep a failed lane alive until every packed lane has written a summary.

    Arnold treats a nonzero role exit as a trial failure and terminates the
    other roles.  A failed short lane must therefore not destroy valid work
    still running in longer lanes.  The shared HDFS mount provides a small
    fail-closed rendezvous without introducing cross-node collectives.
    """

    output_base = Path(str(schedule["output_base"]))
    summary_dir = output_base / str(schedule["stamp"]) / "_packed_workers"
    expected = len(schedule["worker_queues"])
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    last_report = 0.0
    while True:
        complete = sum(
            (summary_dir / f"worker_{index:02d}.json").is_file()
            for index in range(expected)
        )
        if complete == expected:
            print(
                "TRACK3_FINAL_ABLATION_PACK_DRAINED "
                f"worker={worker_index} summaries={complete}/{expected}",
                flush=True,
            )
            return True
        now = time.monotonic()
        if now >= deadline:
            print(
                "TRACK3_FINAL_ABLATION_PACK_DRAIN_TIMEOUT "
                f"worker={worker_index} summaries={complete}/{expected} "
                f"timeout_seconds={timeout_seconds:.1f}",
                flush=True,
            )
            return False
        if now - last_report >= 60:
            print(
                "TRACK3_FINAL_ABLATION_PACK_DRAIN_WAIT "
                f"worker={worker_index} summaries={complete}/{expected}",
                flush=True,
            )
            last_report = now
        time.sleep(min(30.0, max(0.0, deadline - now)))


def main() -> None:
    args = parse_args()
    schedule = json.loads(args.schedule.read_text(encoding="utf-8"))
    queues = schedule["worker_queues"]
    if not 0 <= args.worker_index < len(queues):
        raise SystemExit(
            f"worker index {args.worker_index} outside schedule size {len(queues)}"
        )
    assigned_queue = queues[args.worker_index]
    max_attempts = positive_int_from_env(
        os.environ,
        "TRACK3_CASE_MAX_ATTEMPTS_PER_WORKER",
        DEFAULT_MAX_ATTEMPTS_PER_WORKER,
    )
    print(
        "TRACK3_FINAL_ABLATION_WORKER "
        f"batch={schedule['batch']} index={args.worker_index} "
        f"cases={len(assigned_queue['cases'])} "
        f"estimated_runtime_minutes={assigned_queue['estimated_runtime_minutes']:.2f} "
        f"gpuv={schedule.get('gpuv', 'unknown')} "
        f"max_attempts_per_case={max_attempts}",
        flush=True,
    )
    subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version",
            "--format=csv,noheader",
        ],
        check=False,
    )

    statuses: list[dict[str, object]] = []
    unresolved_case_ids: set[str] = set()
    work_items = collections.deque(
        (position, case, 1)
        for position, case in enumerate(assigned_queue["cases"])
    )
    while work_items:
        position, case, attempt = work_items.popleft()
        env = os.environ.copy()
        env.update({key: str(value) for key, value in case["env"].items()})
        env["HDFS_CODE_TGZ"] = ""
        case_id = env["TRACK3_CASE_ID"]
        already_complete, preflight_timed_out = completed_case_preflight(env)
        if already_complete:
            print(
                "TRACK3_FINAL_ABLATION_SKIP_COMPLETE "
                f"worker={args.worker_index} position={position} "
                f"case_id={case_id} attempt={attempt}",
                flush=True,
            )
            unresolved_case_ids.discard(case_id)
            statuses.append(
                {
                    "case_id": case_id,
                    "position": position,
                    "attempt": attempt,
                    "status": "skipped_complete",
                    "exit_code": 0,
                    "elapsed_seconds": 0.0,
                }
            )
            continue

        archived_evidence = (
            None if preflight_timed_out
            else archive_incomplete_evidence(env, attempt)
        )
        if archived_evidence is not None:
            print(
                "TRACK3_FINAL_ABLATION_ARCHIVE_INCOMPLETE "
                f"worker={args.worker_index} position={position} "
                f"case_id={case_id} attempt={attempt} "
                f"history_dir={archived_evidence}",
                flush=True,
            )
        if local_data_ready():
            env["TRACK3_DATA_HDFS"] = ""
        shutil.rmtree("logs", ignore_errors=True)
        family = str(case["family"])
        entrypoint = ENTRYPOINTS[family]
        started = time.monotonic()
        print(
            "TRACK3_FINAL_ABLATION_CASE_START "
            f"worker={args.worker_index} position={position} "
            f"case_id={case_id} attempt={attempt} max_attempts={max_attempts} "
            f"family={family} recipe={case['recipe']} "
            f"batch={schedule['batch']} coord={case['coord']} "
            f"value={case['value']}",
            flush=True,
        )
        return_code, watchdog_reason = run_with_progress_watchdog(
            ["bash", str(entrypoint)], env
        )
        elapsed = time.monotonic() - started
        persisted = completed_case(env)
        status = "done" if return_code == 0 and persisted else "failed"
        requeued = status == "failed" and attempt < max_attempts
        if status == "done":
            unresolved_case_ids.discard(case_id)
        else:
            unresolved_case_ids.add(case_id)
            if requeued:
                # Put infrastructure failures behind the rest of the initial
                # LPT queue. A persistently broken recipe therefore cannot
                # block unrelated coordinates on the same packed worker.
                work_items.append((position, case, attempt + 1))
        print(
            "TRACK3_FINAL_ABLATION_CASE_END "
            f"worker={args.worker_index} position={position} case_id={case_id} "
            f"attempt={attempt} max_attempts={max_attempts} "
            f"status={status} exit_code={return_code} "
            f"persisted={int(persisted)} elapsed_seconds={elapsed:.1f} "
            f"watchdog_reason={watchdog_reason or 'none'} "
            f"requeued={int(requeued)}",
            flush=True,
        )
        statuses.append(
            {
                "case_id": case_id,
                "position": position,
                "attempt": attempt,
                "status": status,
                "exit_code": return_code,
                "persisted": persisted,
                "elapsed_seconds": elapsed,
                "watchdog_reason": watchdog_reason,
                "requeued": requeued,
            }
        )
        if requeued:
            print(
                "TRACK3_FINAL_ABLATION_CASE_REQUEUE "
                f"worker={args.worker_index} position={position} "
                f"case_id={case_id} next_attempt={attempt + 1} "
                f"remaining_items={len(work_items)}",
                flush=True,
            )

    summary_path = write_worker_summary(schedule, args.worker_index, statuses)
    failed_attempts = [item for item in statuses if item["status"] == "failed"]
    print(
        "TRACK3_FINAL_ABLATION_WORKER_END "
        f"worker={args.worker_index} cases={len(assigned_queue['cases'])} "
        f"attempts={len(statuses)} failed_attempts={len(failed_attempts)} "
        f"unresolved_cases={len(unresolved_case_ids)} summary={summary_path}",
        flush=True,
    )
    if unresolved_case_ids:
        wait_for_pack_drain(
            schedule,
            args.worker_index,
            float(os.environ.get("TRACK3_PACK_DRAIN_TIMEOUT_SECONDS", "7200")),
        )
    raise SystemExit(1 if unresolved_case_ids else 0)


if __name__ == "__main__":
    main()
