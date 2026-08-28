from records.track_3_optimization.batch_size_cd.psgdh_baseline_report import summarize


def test_exact_baseline_gate_checks_all_five_full_seed_terminals():
    rows = [
        {
            "seed": str(seed),
            "status": "DONE",
            "last_step": 3400,
            "total_steps": 3400,
            "last_val_loss": loss,
            "runtime_device_type": "NVIDIA-H20",
        }
        for seed, loss in enumerate(
            (3.27823, 3.27660, 3.27480, 3.27426, 3.27651), start=1
        )
    ]
    report = summarize(rows)
    assert report["passed"] is True
    assert report["mean_loss"] == 3.27608
    assert report["hardware"] == ["NVIDIA-H20"]
