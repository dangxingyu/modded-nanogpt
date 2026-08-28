from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

import pytest

from records.track_3_optimization.batch_size_cd import (
    final_recipe_ablation_report as report,
)
from records.track_3_optimization.batch_size_cd import (
    monitor_final_recipe_ablation as monitor,
)
from records.track_3_optimization.batch_size_cd import (
    run_final_recipe_ablation_worker as worker,
)


def watchdog_env(**overrides: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "TRACK3_CASE_STARTUP_TIMEOUT_SECONDS": "0.1",
            "TRACK3_CASE_STALL_TIMEOUT_SECONDS": "0.1",
            "TRACK3_CASE_POST_TERMINAL_GRACE_SECONDS": "0.1",
            "TRACK3_CASE_TERMINATE_GRACE_SECONDS": "0.1",
            "TRACK3_CASE_CLEANUP_GRACE_SECONDS": "0.1",
            **overrides,
        }
    )
    return env


def terminal_env(tmp_path, stamp: str = "terminal-gate") -> dict[str, str]:
    env = {key: "1" for key in worker.RUNTIME_ENV_KEYS}
    env.update(
        {
            "TRACK3_OUTPUT_BASE": str(tmp_path),
            "TRACK3_STAMP": stamp,
            "TRACK3_CASE_ID": "case-a",
            "TRACK3_RECIPE": "shampooh_core",
            "TRACK3_COORD": "center",
            "TRACK3_TRAIN_STEPS": "813",
        }
    )
    return env


def persist_matching_env(case_dir, env: dict[str, str]) -> None:
    (case_dir / "env.txt").write_text(
        "\n".join(f"{key}={env[key]}" for key in worker.RUNTIME_ENV_KEYS) + "\n"
    )


def test_watchdog_recovers_case_that_never_emits_a_step() -> None:
    started = time.monotonic()
    return_code, reason = worker.run_with_progress_watchdog(
        [sys.executable, "-u", "-c", "import time; time.sleep(30)"],
        watchdog_env(),
    )
    assert return_code != 0
    assert reason == "startup_no_step_timeout"
    assert time.monotonic() - started < 5


def test_initial_validation_does_not_end_compile_startup_budget() -> None:
    return_code, reason = worker.run_with_progress_watchdog(
        [
            sys.executable,
            "-u",
            "-c",
            "import time; print('step:0/10 val_loss:9', flush=True); time.sleep(30)",
        ],
        watchdog_env(
            TRACK3_CASE_STARTUP_TIMEOUT_SECONDS="0.2",
            TRACK3_CASE_STALL_TIMEOUT_SECONDS="0.05",
        ),
    )
    assert return_code != 0
    assert reason == "startup_no_step_timeout"


def test_completed_case_requires_exact_terminal_validation_step(tmp_path) -> None:
    env = terminal_env(tmp_path)
    case_dir = worker.run_dir(env)
    log_dir = case_dir / "logs"
    log_dir.mkdir(parents=True)
    (case_dir / "exit_status.txt").write_text("0\n")
    persist_matching_env(case_dir, env)
    log_path = log_dir / "run.txt"
    log_path.write_text(
        "step:0/813 val_loss:10.8\n"
        "step:500/813 val_loss:3.58\n"
    )

    assert worker.completed_case(env) is False

    with log_path.open("a") as handle:
        handle.write("step:813/813 val_loss:3.33\n")
    assert worker.completed_case(env) is True


def test_completed_case_rejects_exit_zero_with_wrong_horizon(tmp_path) -> None:
    env = terminal_env(tmp_path)
    case_dir = worker.run_dir(env)
    log_dir = case_dir / "logs"
    log_dir.mkdir(parents=True)
    (case_dir / "exit_status.txt").write_text("0\n")
    persist_matching_env(case_dir, env)
    (log_dir / "run.txt").write_text("step:813/1625 val_loss:3.33\n")

    assert worker.completed_case(env) is False


def test_completed_case_rejects_terminal_with_runtime_env_mismatch(tmp_path) -> None:
    env = terminal_env(tmp_path, stamp="env-gate")
    env["TRACK3_COORD"] = "matrix_lr"
    env["TRACK3_MATRIX_LR_MULT"] = "1.4142135623730951"
    case_dir = worker.run_dir(env)
    log_dir = case_dir / "logs"
    log_dir.mkdir(parents=True)
    (case_dir / "exit_status.txt").write_text("0\n")
    persisted = dict(env)
    persisted["TRACK3_MATRIX_LR_MULT"] = "1.0"
    persist_matching_env(case_dir, persisted)
    (log_dir / "run.txt").write_text("step:813/813 val_loss:3.33\n")

    assert worker.completed_case(env) is False


def test_completion_preflight_times_out_hdfs_metadata_lookup(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = terminal_env(tmp_path, stamp="preflight-timeout")
    monkeypatch.setattr(worker, "completed_case", lambda _env: time.sleep(30))
    started = time.monotonic()
    complete, timed_out = worker.completed_case_preflight(
        env, timeout_seconds=0.05
    )
    assert complete is False
    assert timed_out is True
    assert time.monotonic() - started < 2


def test_completion_preflight_retries_transient_hdfs_false_negative(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = terminal_env(tmp_path, stamp="preflight-retry")
    env["TRACK3_COMPLETION_PREFLIGHT_ATTEMPTS"] = "3"
    env["TRACK3_COMPLETION_PREFLIGHT_RETRY_SECONDS"] = "0"
    observations = iter((False, False, True))
    monkeypatch.setattr(worker, "completed_case", lambda _env: next(observations))

    complete, timed_out = worker.completed_case_preflight(
        env, timeout_seconds=1
    )

    assert complete is True
    assert timed_out is False


def test_completion_preflight_refuses_non_main_thread(tmp_path) -> None:
    env = terminal_env(tmp_path, stamp="preflight-thread")
    errors: list[BaseException] = []

    def invoke() -> None:
        try:
            worker.completed_case_preflight(env, timeout_seconds=1)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=invoke)
    thread.start()
    thread.join()
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)


def test_retry_archives_incomplete_evidence_without_deleting_history(tmp_path) -> None:
    env = terminal_env(tmp_path, stamp="archive-gate")
    case_dir = worker.run_dir(env)
    (case_dir / "logs").mkdir(parents=True)
    (case_dir / "logs" / "old.txt").write_text("step:500/813 val_loss:3.58\n")
    (case_dir / "env.txt").write_text("old env\n")
    (case_dir / "exit_status.txt").write_text("0\n")
    (case_dir / "train_materialized.py").write_text("# old\n")
    preserved = case_dir / "attempt_history" / "older_attempt"
    preserved.mkdir(parents=True)
    (preserved / "note.txt").write_text("keep\n")

    archived = worker.archive_incomplete_evidence(env, attempt=2)

    assert archived is not None
    assert (archived / "logs" / "old.txt").is_file()
    assert (archived / "env.txt").is_file()
    assert (archived / "exit_status.txt").is_file()
    assert (archived / "train_materialized.py").is_file()
    assert (preserved / "note.txt").read_text() == "keep\n"
    assert not (case_dir / "logs").exists()
    assert not (case_dir / "env.txt").exists()
    assert not (case_dir / "exit_status.txt").exists()
    assert not (case_dir / "train_materialized.py").exists()


def test_watchdog_recovers_case_whose_step_heartbeat_stops() -> None:
    return_code, reason = worker.run_with_progress_watchdog(
        [
            sys.executable,
            "-u",
            "-c",
            "import time; print('step:1/10', flush=True); time.sleep(30)",
        ],
        watchdog_env(
            TRACK3_CASE_STARTUP_TIMEOUT_SECONDS="5",
        ),
    )
    assert return_code != 0
    assert reason == "step_heartbeat_timeout"


def test_watchdog_forces_python_step_output_unbuffered() -> None:
    return_code, reason = worker.run_with_progress_watchdog(
        [
            sys.executable,
            "-c",
            "import time; print('step:1/10'); time.sleep(30)",
        ],
        watchdog_env(
            TRACK3_CASE_STARTUP_TIMEOUT_SECONDS="5",
        ),
    )
    assert return_code != 0
    assert reason == "step_heartbeat_timeout"


def test_watchdog_preserves_successful_case_exit() -> None:
    return_code, reason = worker.run_with_progress_watchdog(
        [sys.executable, "-u", "-c", "print('step:1/1', flush=True)"],
        watchdog_env(
            TRACK3_CASE_STARTUP_TIMEOUT_SECONDS="5",
            TRACK3_CASE_STALL_TIMEOUT_SECONDS="5",
        ),
    )
    assert return_code == 0
    assert reason is None


def test_post_terminal_hang_is_recovered_only_from_exact_local_log(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    env = terminal_env(tmp_path, stamp="post-terminal-recovery")
    env.update(watchdog_env(
        TRACK3_CASE_STARTUP_TIMEOUT_SECONDS="5",
        TRACK3_CASE_STALL_TIMEOUT_SECONDS="5",
        TRACK3_CASE_POST_TERMINAL_GRACE_SECONDS="0.1",
    ))
    code = (
        "from pathlib import Path; import time; "
        "Path('logs').mkdir(); "
        "Path('logs/run.txt').write_text('step:813/813 val_loss:3.33\\n'); "
        "print('step:813/813 val_loss:3.33', flush=True); time.sleep(30)"
    )

    return_code, reason = worker.run_with_progress_watchdog(
        [sys.executable, "-u", "-c", code], env
    )

    assert return_code == 0
    assert reason == "post_terminal_exit_timeout_recovered"
    assert worker.completed_case(env) is True


def test_post_terminal_stdout_without_exact_local_log_remains_failed(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    env = terminal_env(tmp_path, stamp="post-terminal-fail-closed")
    env.update(watchdog_env(
        TRACK3_CASE_STARTUP_TIMEOUT_SECONDS="5",
        TRACK3_CASE_STALL_TIMEOUT_SECONDS="5",
        TRACK3_CASE_POST_TERMINAL_GRACE_SECONDS="0.1",
    ))
    code = (
        "import time; print('step:813/813 val_loss:3.33', flush=True); "
        "time.sleep(30)"
    )

    return_code, reason = worker.run_with_progress_watchdog(
        [sys.executable, "-u", "-c", code], env
    )

    assert return_code != 0
    assert reason == "post_terminal_exit_timeout"
    assert worker.completed_case(env) is False


def test_monitor_extracts_only_exact_persisted_terminal_marker() -> None:
    stdout = """
TRACK3_FINAL_ABLATION_CASE_START worker=0 position=0 case_id=good attempt=1
step:0/813 val_loss:10.8
step:813/813 val_loss:3.33
TRACK3_FINAL_ABLATION_CASE_END worker=0 position=0 case_id=good attempt=1 status=done exit_code=0 persisted=1 elapsed_seconds=1 watchdog_reason=none requeued=0
TRACK3_FINAL_ABLATION_CASE_START worker=0 position=1 case_id=not-persisted attempt=1
step:813/813 val_loss:3.22
TRACK3_FINAL_ABLATION_CASE_END worker=0 position=1 case_id=not-persisted attempt=1 status=failed exit_code=-9 persisted=0 elapsed_seconds=1 watchdog_reason=x requeued=0
TRACK3_FINAL_ABLATION_CASE_START worker=0 position=2 case_id=partial attempt=1
step:400/813 val_loss:3.88
TRACK3_FINAL_ABLATION_CASE_END worker=0 position=2 case_id=partial attempt=1 status=done exit_code=0 persisted=1 elapsed_seconds=1 watchdog_reason=none requeued=0
"""

    strict = monitor.strict_terminal_cases_from_stdout(stdout, "host-a")

    assert set(strict) == {"good"}
    assert strict["good"]["step"] == strict["good"]["total"] == 813
    assert strict["good"]["terminal_val_loss"] == 3.33
    assert strict["good"]["host"] == "host-a"


def test_successful_case_cleans_orphaned_child_processes() -> None:
    started = time.monotonic()
    code = (
        "import subprocess,sys; "
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
        "print('step:1/1', flush=True)"
    )
    return_code, reason = worker.run_with_progress_watchdog(
        [sys.executable, "-u", "-c", code],
        watchdog_env(
            TRACK3_CASE_STARTUP_TIMEOUT_SECONDS="5",
            TRACK3_CASE_STALL_TIMEOUT_SECONDS="5",
        ),
    )
    assert return_code == 0
    assert reason is None
    assert time.monotonic() - started < 5


def test_worker_requeues_failed_case_and_exits_clean_after_retry(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    schedule_path = tmp_path / "schedule.json"
    output_base = tmp_path / "outputs"
    case = {
        "family": "hyperball",
        "recipe": "shampooh_core",
        "coord": "matrix_lr",
        "value": 1.0,
        "env": {
            "TRACK3_OUTPUT_BASE": str(output_base),
            "TRACK3_STAMP": "retry-test",
            "TRACK3_CASE_ID": "case-a",
        },
    }
    schedule_path.write_text(
        json.dumps(
            {
                "stamp": "retry-test",
                "batch": "2m",
                "output_base": str(output_base),
                "worker_queues": [
                    {"estimated_runtime_minutes": 1.0, "cases": [case]}
                ],
            }
        )
    )
    attempts = 0

    def fake_completed_case(env: dict[str, str]) -> bool:
        return attempts >= 2

    def fake_run(command: list[str], env: dict[str, str]) -> tuple[int, str | None]:
        nonlocal attempts
        attempts += 1
        return ((1, "step_heartbeat_timeout") if attempts == 1 else (0, None))

    monkeypatch.setenv("TRACK3_CASE_MAX_ATTEMPTS_PER_WORKER", "2")
    monkeypatch.setattr(
        worker,
        "parse_args",
        lambda: argparse.Namespace(schedule=schedule_path, worker_index=0),
    )
    monkeypatch.setattr(worker, "completed_case", fake_completed_case)
    monkeypatch.setattr(worker, "local_data_ready", lambda: False)
    monkeypatch.setattr(worker, "run_with_progress_watchdog", fake_run)
    monkeypatch.setattr(worker.subprocess, "run", lambda *args, **kwargs: None)

    with pytest.raises(SystemExit) as exc:
        worker.main()

    assert exc.value.code == 0
    assert attempts == 2
    summary = json.loads(
        (output_base / "retry-test" / "_packed_workers" / "worker_00.json").read_text()
    )
    assert [item["status"] for item in summary["statuses"]] == ["failed", "done"]
    assert [item["attempt"] for item in summary["statuses"]] == [1, 2]
    assert summary["statuses"][0]["requeued"] is True


def test_worker_stops_retrying_after_attempt_limit(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    schedule_path = tmp_path / "schedule.json"
    output_base = tmp_path / "outputs"
    schedule_path.write_text(
        json.dumps(
            {
                "stamp": "retry-limit-test",
                "batch": "2m",
                "output_base": str(output_base),
                "worker_queues": [
                    {
                        "estimated_runtime_minutes": 1.0,
                        "cases": [
                            {
                                "family": "hyperball",
                                "recipe": "shampooh_core",
                                "coord": "matrix_lr",
                                "value": 1.0,
                                "env": {
                                    "TRACK3_OUTPUT_BASE": str(output_base),
                                    "TRACK3_STAMP": "retry-limit-test",
                                    "TRACK3_CASE_ID": "case-a",
                                },
                            }
                        ],
                    }
                ],
            }
        )
    )
    attempts = 0

    def fake_run(command: list[str], env: dict[str, str]) -> tuple[int, str | None]:
        nonlocal attempts
        attempts += 1
        return 1, "startup_no_step_timeout"

    monkeypatch.setenv("TRACK3_CASE_MAX_ATTEMPTS_PER_WORKER", "2")
    monkeypatch.setattr(
        worker,
        "parse_args",
        lambda: argparse.Namespace(schedule=schedule_path, worker_index=0),
    )
    monkeypatch.setattr(worker, "completed_case", lambda env: False)
    monkeypatch.setattr(worker, "local_data_ready", lambda: False)
    monkeypatch.setattr(worker, "run_with_progress_watchdog", fake_run)
    monkeypatch.setattr(worker.subprocess, "run", lambda *args, **kwargs: None)

    with pytest.raises(SystemExit) as exc:
        worker.main()

    assert exc.value.code == 1
    assert attempts == 2
    summary = json.loads(
        (
            output_base
            / "retry-limit-test"
            / "_packed_workers"
            / "worker_00.json"
        ).read_text()
    )
    assert [item["requeued"] for item in summary["statuses"]] == [True, False]


def test_max_attempts_must_be_positive() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        worker.positive_int_from_env(
            {"TRACK3_CASE_MAX_ATTEMPTS_PER_WORKER": "0"},
            "TRACK3_CASE_MAX_ATTEMPTS_PER_WORKER",
            worker.DEFAULT_MAX_ATTEMPTS_PER_WORKER,
        )


def test_failed_lane_waits_for_all_packed_worker_summaries(tmp_path) -> None:
    schedule = {
        "stamp": "drain-test",
        "output_base": str(tmp_path),
        "worker_queues": [{}, {}],
    }
    summary_dir = tmp_path / "drain-test" / "_packed_workers"
    summary_dir.mkdir(parents=True)
    (summary_dir / "worker_00.json").write_text("{}\n")
    assert worker.wait_for_pack_drain(schedule, 0, 0) is False
    (summary_dir / "worker_01.json").write_text("{}\n")
    assert worker.wait_for_pack_drain(schedule, 0, 0) is True


def test_worker_skip_env_contract_matches_report_admission_contract() -> None:
    assert worker.RUNTIME_ENV_KEYS == report.RUNTIME_ENV_KEYS


@pytest.mark.parametrize(
    "line",
    [
        (
            "TRACK3_FINAL_ABLATION_CASE_END worker=0 position=0 case_id=case-a "
            "status=failed exit_code=-15 persisted=0"
        ),
        (
            "TRACK3_FINAL_ABLATION_CASE_END worker=0 position=0 case_id=case-a "
            "attempt=1 max_attempts=3 status=failed exit_code=-15 persisted=0 "
            "requeued=1"
        ),
    ],
)
def test_monitor_parses_legacy_and_retry_case_end_lines(line: str) -> None:
    match = monitor.CASE_END_RE.search(line)
    assert match is not None
    assert match.group("case_id") == "case-a"
    assert match.group("status") == "failed"
    assert int(match.group("exit_code")) == -15
