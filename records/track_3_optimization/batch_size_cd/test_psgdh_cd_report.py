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


def test_build_report_rejects_mixed_hardware(tmp_path):
    round_dir = tmp_path / "round"
    _write_round(round_dir, hardware="H800")
    with pytest.raises(ValueError, match="must all use H20"):
        build_report([round_dir], tmp_path / "report")
