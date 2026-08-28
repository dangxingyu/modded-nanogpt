from __future__ import annotations

import base64
import json
import math
import unittest
from unittest import mock
import zlib

from records.track_3_optimization.batch_size_cd import collect_core_cd as collect


def child(case_id: str, coord: str) -> dict[str, str]:
    return {
        "TRACK3_STAMP": "pretraining_wrapper_test",
        "TRACK3_CASE_ID": case_id,
        "TRACK3_RECIPE": "lionw_pr326_core",
        "TRACK3_BATCH_SIZE": "131072",
        "TRACK3_TRAIN_STEPS": "13000",
        "TRACK3_COORD": coord,
        "TRACK3_MATRIX_LR_MULT": "1",
        "TRACK3_AUX_LR_MULT": "1",
        "TRACK3_MATRIX_BETA1_OM_MULT": "1",
        "TRACK3_MATRIX_BETA2_OM_MULT": "1",
        "TRACK3_AUX_BETA1_OM_MULT": "1",
        "TRACK3_AUX_BETA2_OM_MULT": "1",
        "TRACK3_MATRIX_WD_PEAK": "6",
        "TRACK3_AUX_WD_PEAK": "4",
        "TRACK3_SEED": "1",
        "TRACK3_HARDWARE_FAMILY": "H20",
    }


def wrapper_item(cases: list[dict[str, str]], status: str = "RUNNING") -> dict:
    encoded = base64.b64encode(
        zlib.compress(json.dumps(cases).encode("utf-8"))
    ).decode("ascii")
    return {
        "id": "0123456789abcdef",
        "status": status,
        "latest_trial_id": "trial",
        "job_def_name": "pretraining_wrapper_test-wrapper_00",
        "meta": {
            "job_def_version": {
                "env": {
                    **cases[0],
                    "TRACK3_CASE_ID": "wrapper_00",
                    "TRACK3_COORD": "wrapper",
                    "TRACK3_WRAPPER_CASES_ZLIB_B64": encoded,
                },
                "image_meta": {"image_url": "test-image"},
                "resource": {
                    "arnold_config": {
                        "clusterName": "palm-wlby",
                        "roles": [
                            {
                                "gpuv": "NVIDIA_H20",
                                "queueName": "nvidia-h20.test.ai",
                            }
                        ],
                    }
                },
            }
        },
    }


class CollectWrapperTest(unittest.TestCase):
    def test_injects_local_manifest_into_embedded_schedule_job(self) -> None:
        cases = [child("child-a", "center"), child("child-b", "matrix_lr")]
        item = {"meta": {"job_def_version": {"env": {"HDFS_CODE_TGZ": "x"}}}}
        collect.inject_packed_manifest(item, cases)
        env = collect.get_env(item)
        self.assertEqual(env["TRACK3_COORD"], "wrapper")
        self.assertEqual(collect.decode_wrapper_cases(env), cases)
        self.assertEqual(env["HDFS_CODE_TGZ"], "x")

    @mock.patch.object(collect, "run_merlin")
    def test_list_run_query_failure_does_not_retry_same_page_forever(
        self, run_merlin: mock.Mock
    ) -> None:
        run_merlin.side_effect = RuntimeError("malformed Merlin response")
        stamp = "pretraining_test_collect_failure"
        self.assertEqual(collect.list_runs(stamp, 100), [])
        self.assertEqual(run_merlin.call_count, len(collect.stamp_query_terms(stamp)))

    def test_nccl_startup_information_is_not_a_failure(self) -> None:
        parsed = collect.parse_logs(
            "NCCL version 2.27.3+cuda12.9\n"
            "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800\n"
            "step:10/100 val_loss:4.2\n"
        )
        self.assertEqual(parsed["failure_kind"], "")

    def test_explicit_nccl_errors_are_node_failures(self) -> None:
        for text in (
            "torch.distributed.DistBackendError: NCCL error in collective",
            "Watchdog caught collective operation timeout",
            "ncclSystemError: unhandled system error",
        ):
            with self.subTest(text=text):
                self.assertEqual(collect.parse_logs(text)["failure_kind"], "nccl_or_node")

    def test_decodes_compressed_and_legacy_wrapper_payloads(self) -> None:
        cases = [child("child-a", "center"), child("child-b", "matrix_lr")]
        compressed = wrapper_item(cases)["meta"]["job_def_version"]["env"]
        self.assertEqual(collect.decode_wrapper_cases(compressed), cases)

        legacy = {
            "TRACK3_WRAPPER_CASES_B64": base64.b64encode(
                json.dumps(cases).encode("utf-8")
            ).decode("ascii")
        }
        self.assertEqual(collect.decode_wrapper_cases(legacy), cases)

    def test_child_status_comes_from_child_artifacts(self) -> None:
        self.assertEqual(collect.child_status("RUNNING", "0\n", True), "DONE")
        self.assertEqual(collect.child_status("RUNNING", "9\n", True), "FAILED")
        self.assertEqual(collect.child_status("RUNNING", "", True), "RUNNING")
        self.assertEqual(collect.child_status("RUNNING", "", False), "PENDING")
        self.assertEqual(collect.child_status("FAILED", "", False), "NOT_STARTED")

    @mock.patch.object(collect, "fetch_merlin_text")
    @mock.patch.object(collect, "read_run_dir_exit_status")
    @mock.patch.object(collect, "read_run_dir_logs")
    def test_parent_loss_never_fills_an_unstarted_child(
        self,
        read_logs: mock.Mock,
        read_exit: mock.Mock,
        fetch_parent: mock.Mock,
    ) -> None:
        cases = [child("child-a", "center"), child("child-b", "matrix_lr")]
        read_logs.side_effect = [
            ("step:13000/13000 val_loss:3.000000", "TRACK3_CASE_ID=child-a\n"),
            ("", ""),
        ]
        read_exit.side_effect = ["0\n", ""]
        fetch_parent.return_value = "step:13000/13000 val_loss:1.000000"

        rows = collect.rows_from_item(
            "pretraining_wrapper_test",
            wrapper_item(cases),
            fetch_logs=True,
            probe_hdfs=True,
            timeout=1,
        )

        self.assertEqual([row["case_id"] for row in rows], ["child-a", "child-b"])
        self.assertEqual(rows[0]["status"], "DONE")
        self.assertEqual(rows[0]["last_val_loss"], 3.0)
        self.assertEqual(rows[1]["status"], "PENDING")
        self.assertIsNone(rows[1]["last_val_loss"])
        self.assertEqual(rows[1]["num_val_points"], 0)

    @mock.patch.object(collect, "fetch_merlin_text")
    @mock.patch.object(collect, "read_run_dir_exit_status")
    @mock.patch.object(collect, "read_run_dir_logs")
    def test_zero_exit_status_does_not_accept_an_interrupted_child(
        self,
        read_logs: mock.Mock,
        read_exit: mock.Mock,
        fetch_parent: mock.Mock,
    ) -> None:
        cases = [child("child-a", "center")]
        read_logs.return_value = (
            "step:5/13000 val_loss:10.82584",
            "TRACK3_CASE_ID=child-a\n",
        )
        read_exit.return_value = "0\n"
        fetch_parent.return_value = "NCCL version 2.27.3+cuda12.9"

        rows = collect.rows_from_item(
            "pretraining_wrapper_test",
            wrapper_item(cases, status="STOPPED"),
            fetch_logs=True,
            probe_hdfs=True,
            timeout=1,
        )

        self.assertEqual(rows[0]["status"], "FAILED")
        self.assertEqual(rows[0]["failure_kind"], "incomplete_artifact")
        self.assertEqual(rows[0]["last_step"], 5)

    @mock.patch.object(collect, "fetch_merlin_text")
    @mock.patch.object(collect, "read_run_dir_exit_status")
    @mock.patch.object(collect, "read_run_dir_logs")
    def test_nonfinite_terminal_is_complete_scientific_divergence(
        self,
        read_logs: mock.Mock,
        read_exit: mock.Mock,
        fetch_parent: mock.Mock,
    ) -> None:
        cases = [child("child-a", "matrix_lr")]
        read_logs.return_value = (
            "step:13000/13000 val_loss:nan",
            "TRACK3_CASE_ID=child-a\n",
        )
        read_exit.return_value = "0\n"
        fetch_parent.return_value = ""

        rows = collect.rows_from_item(
            "pretraining_wrapper_test",
            wrapper_item(cases, status="DONE"),
            fetch_logs=True,
            probe_hdfs=True,
            timeout=1,
        )

        self.assertEqual(rows[0]["status"], "DONE")
        self.assertTrue(math.isnan(rows[0]["last_val_loss"]))
        self.assertEqual(rows[0]["failure_kind"], "nan_or_divergence")
        self.assertEqual(collect.terminal_losses(rows), [])


if __name__ == "__main__":
    unittest.main()
