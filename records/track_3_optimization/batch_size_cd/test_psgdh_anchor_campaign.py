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


def test_schedule_is_packed_and_balanced():
    cases = campaign.build_cases(args())
    schedule = campaign.schedule_cases(cases, workers=8)
    assert schedule["case_count"] == 32
    assert len(schedule["worker_queues"]) == 8
    counts = sorted(len(queue["cases"]) for queue in schedule["worker_queues"])
    assert counts == [4] * 8


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
