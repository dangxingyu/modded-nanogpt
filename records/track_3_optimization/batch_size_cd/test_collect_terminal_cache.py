from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from records.track_3_optimization.batch_size_cd import collect_core_cd as collect


class CollectTerminalCacheTest(unittest.TestCase):
    def test_only_terminal_rows_are_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "collect.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "job_run_id",
                        "status",
                        "train_steps",
                        "last_step",
                        "last_val_step",
                        "last_val_loss",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "job_run_id": "done",
                        "status": "DONE",
                        "train_steps": "100",
                        "last_step": "100",
                        "last_val_step": "100",
                        "last_val_loss": "3.1",
                    }
                )
                writer.writerow(
                    {
                        "job_run_id": "truncated",
                        "status": "DONE",
                        "train_steps": "100",
                        "last_step": "5",
                        "last_val_step": "5",
                        "last_val_loss": "10.8",
                    }
                )
                writer.writerow(
                    {
                        "job_run_id": "failed-before-robust-retry",
                        "status": "FAILED",
                        "train_steps": "100",
                        "last_step": "5",
                        "last_val_step": "5",
                        "last_val_loss": "10.8",
                    }
                )
                writer.writerow({"job_run_id": "running", "status": "RUNNING", "last_val_loss": ""})
            cache = collect.load_terminal_cache(path)
        self.assertEqual(set(cache), {("done", "")})
        self.assertEqual(cache[("done", "")]["last_val_loss"], "3.1")

    def test_missing_cache_is_empty(self) -> None:
        self.assertEqual(collect.load_terminal_cache(Path("/definitely/missing.csv")), {})

    def test_compact_recovery_caption_is_queried(self) -> None:
        stamp = "pretraining_h20_recovery_wave226_mlr0707_20260718_2338"
        terms = collect.stamp_query_terms(stamp)
        self.assertIn("pretraining_h20_w226_mlr0707_20260718_2338", terms)
        self.assertIn("pretraining_h20_w226_", terms)

    def test_seed_us_recovery_caption_uses_stable_wave_prefix(self) -> None:
        stamp = (
            "pretraining_h20_seed_us_recovery_wave224_"
            "matrix_wd_peak_0p15_20260718_2324"
        )
        self.assertIn("pretraining_h20_w224_", collect.stamp_query_terms(stamp))

    def test_compact_diagnostic_caption_is_queried(self) -> None:
        stamp = "pretraining_h800_diagnostic_wave228_mwd015_20260718_2355"
        terms = collect.stamp_query_terms(stamp)
        self.assertIn("pretraining_h800_w228_mwd015_20260718_2355", terms)
        self.assertIn("pretraining_h800_w228_", terms)

    def test_same_value_accepts_legacy_rounded_centers(self) -> None:
        self.assertTrue(collect.same_value(0.5, 0.353553 * 2**0.5))
        self.assertTrue(collect.same_value(2.0, 1.41421 * 2**0.5))
        self.assertFalse(collect.same_value(0.5, 0.49))

    def test_get_run_item_returns_job_metadata(self) -> None:
        original = collect.run_merlin
        try:
            collect.run_merlin = lambda command, payload: {
                "job_run": {"id": payload["job_run_id"], "status": "RUNNING"}
            }
            item = collect.get_run_item("0123456789abcdef")
        finally:
            collect.run_merlin = original
        self.assertEqual(item["id"], "0123456789abcdef")
        self.assertEqual(item["status"], "RUNNING")

if __name__ == "__main__":
    unittest.main()
