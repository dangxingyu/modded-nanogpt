import json

import pytest

from records.track_3_optimization.batch_size_cd.psgdh_cd_report import build_report


def _write_round(path, batch="512k", hardware="H20"):
    path.mkdir()
    manifest = []
    collected = []
    for index in range(32):
        case_id = f"case-{index}"
        manifest.append(
            {
                "case_id": case_id,
                "batch": batch,
                "env": {"TRACK3_STAMP": f"stamp-{batch}"},
            }
        )
        collected.append(
            {
                "case_id": case_id,
                "stamp": f"stamp-{batch}",
                "batch": batch,
                "coord": "center" if index == 0 else "matrix_lr",
                "status": "DONE",
                "last_step": 10,
                "total_steps": 10,
                "last_val_loss": 3.0 + index / 1000,
                "hardware_family": hardware,
                "gpuv": "NVIDIA_H20",
            }
        )
    selection = {
        "stamp": f"stamp-{batch}",
        "batch": batch,
        "center_case_id": "case-0",
        "center_loss": 3.0,
        "center_env": {"TRACK3_MATRIX_LR_MULT": "1"},
        "evidence": {
            "matrix_lr": {
                "winner_case_id": "case-0",
                "raw_best_case_id": "case-0",
                "raw_improvement": 0.0,
                "improvement": 0.0,
                "boundary": False,
            }
        },
        "converged": True,
        "needs_boundary_extension": False,
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    (path / "collected.json").write_text(json.dumps(collected))
    (path / "selection.json").write_text(json.dumps(selection))


def test_build_report_writes_credential_free_tables(tmp_path):
    round_dir = tmp_path / "round"
    _write_round(round_dir)
    output = tmp_path / "report"
    result = build_report([round_dir], output)
    assert result["batches"] == ["512k"]
    assert "job_run_id" not in (output / "raw_results.csv").read_text()
    assert "512k" in (output / "REPORT.md").read_text()
    assert "Matrix LR factor" in (output / "REPORT.md").read_text()
    assert (output / "coordinate_improvements.csv").is_file()
    assert (output / "center_loss_by_round.svg").is_file()


def test_build_report_rejects_mixed_hardware(tmp_path):
    round_dir = tmp_path / "round"
    _write_round(round_dir, hardware="H800")
    with pytest.raises(ValueError, match="must all use H20"):
        build_report([round_dir], tmp_path / "report")


def test_build_report_orders_batches_by_token_count(tmp_path):
    round_dirs = []
    for batch in ("2m", "512k", "128k", "1m"):
        round_dir = tmp_path / batch
        _write_round(round_dir, batch=batch)
        round_dirs.append(round_dir)
    result = build_report(round_dirs, tmp_path / "report")
    assert result["batches"] == ["128k", "512k", "1m", "2m"]


def test_build_report_prefers_recovery_merged_rows(tmp_path):
    round_dir = tmp_path / "round"
    _write_round(round_dir)
    rows = json.loads((round_dir / "collected.json").read_text())
    rows[0]["last_val_loss"] = 2.9
    (round_dir / "collected_merged.json").write_text(json.dumps(rows))

    build_report([round_dir], tmp_path / "report")

    raw = (tmp_path / "report" / "raw_results.csv").read_text()
    assert "2.9" in raw


def test_build_report_prefers_complete_rows_over_earlier_snapshots(tmp_path):
    round_dir = tmp_path / "round"
    _write_round(round_dir)
    rows = json.loads((round_dir / "collected.json").read_text())
    rows[0]["last_val_loss"] = 2.8
    (round_dir / "collected_merged.json").write_text(json.dumps(rows))
    rows[0]["last_val_loss"] = 2.7
    (round_dir / "collected_complete.json").write_text(json.dumps(rows))

    build_report([round_dir], tmp_path / "report")

    raw = (tmp_path / "report" / "raw_results.csv").read_text()
    assert "2.7" in raw
    assert "2.8" not in raw
