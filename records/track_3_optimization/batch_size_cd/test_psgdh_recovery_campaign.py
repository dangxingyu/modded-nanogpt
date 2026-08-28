from records.track_3_optimization.batch_size_cd import psgdh_recovery_campaign as recovery


def _case(index: int, batch: str = "512k") -> dict:
    return {
        "case_id": f"case-{index}",
        "batch": batch,
        "train_steps": 10,
        "env": {
            "TRACK3_CASE_ID": f"case-{index}",
            "TRACK3_STAMP": "source",
            "TRACK3_OUTPUT_BASE": "/tmp/results",
        },
    }


def test_is_resolved_requires_full_done_terminal() -> None:
    assert recovery.is_resolved(
        {
            "case_id": "case-0",
            "status": "DONE",
            "last_step": 10,
            "total_steps": 10,
            "last_val_loss": 3.2,
        }
    )
    assert not recovery.is_resolved(
        {
            "case_id": "case-0",
            "status": "FAILED",
            "last_step": 10,
            "total_steps": 10,
            "last_val_loss": 3.2,
        }
    )
    assert not recovery.is_resolved(
        {
            "case_id": "case-0",
            "status": "DONE",
            "last_step": 9,
            "total_steps": 10,
            "last_val_loss": 3.2,
        }
    )


def test_unresolved_cases_preserve_identity_and_science() -> None:
    manifest = [_case(0), _case(1)]
    rows = [
        {
            "case_id": "case-0",
            "status": "DONE",
            "last_step": 10,
            "total_steps": 10,
            "last_val_loss": 3.2,
        }
    ]
    cases = recovery.unresolved_cases(manifest, rows, "recovery")
    assert [case["case_id"] for case in cases] == ["case-1"]
    assert cases[0]["env"]["TRACK3_CASE_ID"] == "case-1"
    assert cases[0]["env"]["TRACK3_STAMP"] == "recovery"
    assert cases[0]["env"]["TRACK3_CASE_STARTUP_TIMEOUT_SECONDS"] == "10800"
    assert cases[0]["env"]["TRACK3_CASE_STALL_TIMEOUT_SECONDS"] == "900"
    assert cases[0]["env"]["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] == "10800"
    assert manifest[1]["env"]["TRACK3_STAMP"] == "source"


def test_single_case_recovery_still_packs_one_32_gpu_job() -> None:
    schedule = recovery.schedule_recovery_cases([_case(0)], workers=4)
    assert schedule["worker_count"] == 4
    assert schedule["case_count"] == 1
    assert [len(queue["cases"]) for queue in schedule["worker_queues"]] == [1, 0, 0, 0]
