from types import SimpleNamespace

from records.track_3_optimization.batch_size_cd import materialize_core_hparam
from records.track_3_optimization.batch_size_cd import psgdh_pr316_baseline_campaign as campaign


def args(workers=5):
    return SimpleNamespace(
        stamp="pretraining_test_psgdh_pr316_baseline",
        workers=workers,
        track3_data_hdfs="hdfs://example/data",
        output_base="/mnt/hdfs/example/runs",
    )


def test_exact_recipe_is_unmodified_pr316_source():
    expected = (
        materialize_core_hparam.ROOT
        / materialize_core_hparam.RECIPES["psgdh_pr316_exact"]["path"]
    ).read_text()
    assert materialize_core_hparam.materialize("psgdh_pr316_exact") == expected


def test_baseline_repetitions_are_packed_one_per_worker():
    cases = campaign.build_cases(args())
    schedule = campaign.schedule_cases(cases)
    assert len(cases) == 5
    assert schedule["worker_count"] == 5
    assert schedule["case_count"] == 5
    assert all(len(queue["cases"]) == 1 for queue in schedule["worker_queues"])
    assert all(case["recipe"] == "psgdh_pr316_exact" for case in cases)
    assert all(case["train_steps"] == 3400 for case in cases)


def test_baseline_enforces_minimum_32_gpu_pack():
    try:
        campaign.build_cases(args(workers=3))
    except SystemExit as exc:
        assert "at least 32" in str(exc)
    else:
        raise AssertionError("expected a sub-32-GPU pack to be rejected")
