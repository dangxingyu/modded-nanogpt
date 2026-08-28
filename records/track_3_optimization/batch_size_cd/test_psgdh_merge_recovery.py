from __future__ import annotations

import pytest

from records.track_3_optimization.batch_size_cd.psgdh_merge_recovery import (
    merge_rows,
)


def row(case_id: str, status: str, step: int, loss: float | None) -> dict:
    return {
        "case_id": case_id,
        "status": status,
        "last_step": step,
        "total_steps": 10,
        "last_val_loss": loss,
    }


def test_recovery_replaces_incomplete_base_row_in_manifest_order() -> None:
    manifest = [{"case_id": "a"}, {"case_id": "b"}]
    base = [row("a", "DONE", 10, 3.2), row("b", "FAILED", 4, None)]
    recovery = [row("b", "DONE", 10, 3.1)]

    merged = merge_rows(manifest, [base, recovery])

    assert [item["case_id"] for item in merged] == ["a", "b"]
    assert [item["last_val_loss"] for item in merged] == [3.2, 3.1]


def test_merge_refuses_any_remaining_incomplete_case() -> None:
    manifest = [{"case_id": "a"}, {"case_id": "b"}]
    with pytest.raises(ValueError, match="incomplete collected rows"):
        merge_rows(
            manifest,
            [[row("a", "DONE", 10, 3.2), row("b", "FAILED", 4, None)]],
        )
