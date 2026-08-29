from __future__ import annotations

from records.track_3_optimization.batch_size_cd import (
    psgdh_recover_strict_terminal as recovery,
)


def row(*, status: str = "FAILED", step: int | None = None) -> dict[str, object]:
    return {
        "case_id": "case-a",
        "status": status,
        "failure_kind": "incomplete_artifact" if status != "DONE" else "",
        "last_step": step,
        "total_steps": step,
        "last_val_loss": 3.5 if step else None,
        "train_steps": 10,
        "log_source": "run_dir_metadata",
    }


def exact_evidence(*, total: int = 10) -> dict[str, object]:
    return {
        "case_id": "case-a",
        "step": total,
        "total": total,
        "terminal_val_loss": 3.25,
        "host": "host-a",
        "evidence": "case_end_done_exit0_persisted1",
    }


def test_recovers_only_exact_persisted_terminal_evidence() -> None:
    rows = recovery.recover_rows(
        [row()], [("trial-old", {"case-a": exact_evidence()})]
    )
    assert rows[0]["status"] == "DONE"
    assert rows[0]["last_step"] == rows[0]["total_steps"] == 10
    assert rows[0]["last_val_loss"] == 3.25
    assert rows[0]["trial_id"] == "trial-old"
    assert rows[0]["log_source"] == "merlin_strict_terminal"


def test_does_not_overwrite_an_existing_complete_row() -> None:
    rows = recovery.recover_rows(
        [row(status="DONE", step=10)],
        [("trial-old", {"case-a": exact_evidence()})],
    )
    assert rows[0]["last_val_loss"] == 3.5
    assert rows[0]["log_source"] == "run_dir_metadata"


def test_rejects_terminal_evidence_for_the_wrong_horizon() -> None:
    rows = recovery.recover_rows(
        [row()], [("trial-old", {"case-a": exact_evidence(total=9)})]
    )
    assert rows[0]["status"] == "FAILED"


def test_recovers_exact_persisted_terminal_divergence() -> None:
    evidence = exact_evidence()
    evidence["terminal_val_loss"] = None
    evidence["failure_kind"] = "nan_or_divergence"

    rows = recovery.recover_rows(
        [row()], [("trial-old", {"case-a": evidence})]
    )

    assert rows[0]["status"] == "DONE"
    assert rows[0]["last_step"] == rows[0]["total_steps"] == 10
    assert rows[0]["last_val_loss"] is None
    assert rows[0]["failure_kind"] == "nan_or_divergence"
