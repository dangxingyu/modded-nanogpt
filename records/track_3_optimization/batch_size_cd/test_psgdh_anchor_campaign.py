from types import SimpleNamespace

from records.track_3_optimization.batch_size_cd import psgdh_anchor_campaign as campaign


def args():
    return SimpleNamespace(
        stamp="pretraining_test_psgdh",
        token_horizon=3250 * 512 * 1024,
        seed=1,
        track3_data_hdfs="hdfs://example/data",
        output_base="/mnt/hdfs/example/runs",
        batch="512k",
        center_env=[],
    )


def test_anchor_sweep_has_one_center_and_all_one_entry_arms():
    cases = campaign.build_cases(args())
    assert len(cases) == 32
    assert sum(case["coord"] == "center" for case in cases) == 1
    for case in cases:
        changed = {
            key
            for key, value in case["env"].items()
            if key in campaign.BASE_ENV and value != campaign.BASE_ENV[key]
        }
        if case["coord"] == "center":
            assert changed == set()
        else:
            expected = next(
                env_key
                for coord, env_key, _ in campaign.COORDINATES
                if coord == case["coord"]
            )
            assert changed == {expected}


def test_anchor_cases_allow_cold_psgd_compile_to_finish():
    cases = campaign.build_cases(args())
    assert {
        case["env"]["TRACK3_CASE_STARTUP_TIMEOUT_SECONDS"] for case in cases
    } == {"10800"}
    assert {
        case["env"]["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] for case in cases
    } == {"10800"}
    assert {
        case["env"]["TRACK3_CASE_STALL_TIMEOUT_SECONDS"] for case in cases
    } == {"900"}
    assert {
        case["env"]["TRACK3_CASE_COMPILE_GRACE_STEPS"] for case in cases
    } == {"120"}
    assert {
        case["env"]["TRACK3_STRICT_COLLECTIVE_COMPLETION"] for case in cases
    } == {"0"}
    assert {
        case["env"]["TRACK3_GRADIENT_COLLECTIVE_COMPLETION"] for case in cases
    } == {"0"}
    assert {
        case["env"]["TRACK3_GRADIENT_PHASE_COMPLETION"] for case in cases
    } == {"0"}
    assert all("TRACK3_EXPLICIT_GRADIENT_WORKS" not in case["env"] for case in cases)
    assert all(
        "TRACK3_PSGD_EXPLICIT_GATHER_COMPLETION" not in case["env"] for case in cases
    )
    assert {
        case["env"]["TRACK3_OPTIMIZER_STEP_COMPLETION"] for case in cases
    } == {"0"}
    assert {
        case["env"]["TRACK3_OPTIMIZER_PHASE_BARRIER"] for case in cases
    } == {"1"}


def test_schedule_is_packed_and_balanced():
    cases = campaign.build_cases(args())
    schedule = campaign.schedule_cases(cases, workers=8)
    assert schedule["case_count"] == 32
    assert len(schedule["worker_queues"]) == 8
    counts = sorted(len(queue["cases"]) for queue in schedule["worker_queues"])
    assert counts == [4] * 8


def test_fixed_token_runtime_estimate_has_large_batch_floor():
    value = args()
    value.batch = "2m"
    schedule = campaign.schedule_cases(campaign.build_cases(value), workers=8)
    case_minutes = [
        case["estimated_runtime_minutes"]
        for queue in schedule["worker_queues"]
        for case in queue["cases"]
    ]
    assert set(case_minutes) == {30.0}
    assert schedule["worker_queues"][0]["estimated_runtime_minutes"] == 120.0


def test_unguaranteed_quota_is_an_explicit_payload_mode():
    payload = {"resource_config": {"arnold_config": {"roles": [{}]}}}
    campaign.apply_quota_mode(payload, False)
    assert "advanced" not in payload["resource_config"]["arnold_config"]["roles"][0]
    campaign.apply_quota_mode(payload, True)
    assert payload["resource_config"]["arnold_config"]["roles"][0]["advanced"] == {
        "is_use_unguaranteed_quota": True
    }


def test_explicit_unlisted_parent_group_preserves_exact_route(monkeypatch):
    monkeypatch.setattr(
        campaign.packed,
        "live_resource_check",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            SystemExit("Expected one exact live resource row, found 0")
        ),
    )
    value = SimpleNamespace(
        allow_unlisted_resource=True,
        group_id=2145,
        cluster_id=48,
        queue_name="h20-parent-queue",
        gpuv="NVIDIA_H20",
    )
    evidence = campaign.live_resource_evidence(value, 256)
    assert evidence["group_sid"] == "2145"
    assert evidence["cluster_sid"] == "48"
    assert evidence["queue_name"] == "h20-parent-queue"
    assert evidence["requested_gpus"] == 256
    assert evidence["query_scope"] == "explicitly_authorized_unlisted_parent_group"


def test_batch_and_center_are_explicitly_reusable_for_later_rounds():
    value = args()
    value.batch = "1m"
    value.center_env = ["TRACK3_MATRIX_LR_MULT=1.41421356237"]
    cases = campaign.build_cases(value)
    assert {case["batch"] for case in cases} == {"1m"}
    assert {case["train_steps"] for case in cases} == {1625}
    center = next(case for case in cases if case["coord"] == "center")
    assert center["env"]["TRACK3_MATRIX_LR_MULT"] == "1.41421356237"
    high = next(
        case
        for case in cases
        if case["coord"] == "matrix_lr" and case["factor"] == 2**0.5
    )
    assert abs(float(high["env"]["TRACK3_MATRIX_LR_MULT"]) - 2.0) < 1e-10


def test_cooldown_grids_move_with_the_current_center_and_keep_both_sides():
    value = args()
    value.center_env = [
        "TRACK3_AUX_COOLDOWN_FRAC=0.8",
        "TRACK3_H_COOLDOWN_FRAC=0.8",
    ]
    cases = campaign.build_cases(value)
    assert len(cases) == 32
    aux_values = {
        float(case["env"]["TRACK3_AUX_COOLDOWN_FRAC"])
        for case in cases
        if case["coord"] == "aux_cooldown_frac"
    }
    matrix_values = {
        float(case["env"]["TRACK3_H_COOLDOWN_FRAC"])
        for case in cases
        if case["coord"] == "matrix_cooldown_frac"
    }
    assert aux_values == {0.0, 0.2, 0.4, 0.6, 1.0}
    assert matrix_values == {0.6, 1.0}
