from types import SimpleNamespace

from records.track_3_optimization.batch_size_cd import psgdh_anchor_campaign
from records.track_3_optimization.batch_size_cd.psgdh_cd_select import select_round


def test_select_round_combines_each_coordinate_winner():
    args = SimpleNamespace(
        stamp="pretraining_test_psgdh_select",
        token_horizon=3250 * 512 * 1024,
        seed=1,
        track3_data_hdfs="hdfs://example/data",
        output_base="/mnt/hdfs/example/runs",
        batch="512k",
        center_env=[],
    )
    manifest = psgdh_anchor_campaign.build_cases(args)
    rows = []
    for case in manifest:
        loss = 3.0
        if case["coord"] != "center" and case.get("factor") == 2**0.5:
            loss = 2.9
        rows.append(
            {
                "case_id": case["case_id"],
                "status": "DONE",
                "last_step": case["train_steps"],
                "total_steps": case["train_steps"],
                "last_val_loss": loss,
            }
        )
    selected = select_round(manifest, rows)
    assert selected["center_loss"] == 3.0
    assert selected["center_env"]["TRACK3_MATRIX_LR_MULT"] == "1.41421356237"
    assert selected["center_env"]["TRACK3_PRECOND_LR_MULT"] == "1.41421356237"
    assert selected["needs_boundary_extension"] is False


def test_select_round_treats_nonfinite_arm_as_resolved_negative_result():
    args = SimpleNamespace(
        stamp="pretraining_test_psgdh_nonfinite",
        token_horizon=3250 * 512 * 1024,
        seed=1,
        track3_data_hdfs="hdfs://example/data",
        output_base="/mnt/hdfs/example/runs",
        batch="512k",
        center_env=[],
    )
    manifest = psgdh_anchor_campaign.build_cases(args)
    rows = []
    divergent = None
    for case in manifest:
        loss = 3.0
        if case["coord"] == "precond_lr" and case["factor"] == 2.0:
            loss = float("nan")
            divergent = case["case_id"]
        rows.append(
            {
                "case_id": case["case_id"],
                "status": "DONE",
                "last_step": case["train_steps"],
                "total_steps": case["train_steps"],
                "last_val_loss": loss,
            }
        )
    selected = select_round(manifest, rows)
    assert selected["center_env"]["TRACK3_PRECOND_LR_MULT"] == "1"
    assert selected["evidence"]["precond_lr"]["scientific_failure_case_ids"] == [
        divergent
    ]


def test_select_round_keeps_subthreshold_move_as_raw_evidence():
    args = SimpleNamespace(
        stamp="pretraining_test_psgdh_threshold",
        token_horizon=3250 * 512 * 1024,
        seed=1,
        track3_data_hdfs="hdfs://example/data",
        output_base="/mnt/hdfs/example/runs",
        batch="512k",
        center_env=[],
    )
    manifest = psgdh_anchor_campaign.build_cases(args)
    rows = []
    raw_best_case_id = ""
    for case in manifest:
        loss = 3.01 if case["coord"] != "center" else 3.0
        if case["coord"] == "matrix_lr" and case["factor"] == 2**0.5:
            loss = 2.999
            raw_best_case_id = case["case_id"]
        rows.append(
            {
                "case_id": case["case_id"],
                "status": "DONE",
                "last_step": case["train_steps"],
                "total_steps": case["train_steps"],
                "last_val_loss": loss,
            }
        )
    selected = select_round(manifest, rows, min_improvement=0.003)
    evidence = selected["evidence"]["matrix_lr"]
    assert selected["center_env"]["TRACK3_MATRIX_LR_MULT"] == "1"
    assert evidence["winner_case_id"] == selected["center_case_id"]
    assert evidence["raw_best_case_id"] == raw_best_case_id
    assert 0 < evidence["raw_improvement"] < 0.003
