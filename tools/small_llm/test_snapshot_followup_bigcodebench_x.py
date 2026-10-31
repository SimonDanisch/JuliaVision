import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from snapshot_bigcodebench_x import digest, SELECTION_SEED
from snapshot_followup_bigcodebench_x import reserve


class FollowupSnapshotTests(unittest.TestCase):
    def parent(self, directory):
        directory.mkdir()
        records = [{"task_id": f"External/{i}", "prompt": "public prompt", "program": "print('reference')", "test_suite": "import json\ndef test_cases(runner): pass"} for i in range(20)]
        ordered = sorted(records, key=lambda row: digest((SELECTION_SEED + "\0" + row["task_id"]).encode()))
        pq.write_table(pa.Table.from_pylist(records), directory / "dataset.parquet")
        packet = {"dataset": "fixture", "revision": "fixture", "repository_commit": "fixture", "reference_language": "Python", "selected_count": 8, "selected_ids": [r["task_id"] for r in ordered[:8]], "total_records": 20, "selection_seed": SELECTION_SEED, "files_sha256": {"dataset.parquet": digest((directory / "dataset.parquet").read_bytes())}}
        (directory / "manifest.json").write_text(json.dumps(packet))
        return ordered

    def test_next_eight_and_public_private_separation(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root);ordered = self.parent(root / "parent")
            report = reserve(root / "parent", root / "followup")
            self.assertEqual(report["selected_ids"], [r["task_id"] for r in ordered[8:16]])
            self.assertFalse(set(report["selected_ids"]) & set(report["excluded_prior_ids"]))
            public = [json.loads(line) for line in (root / "followup/public_prompts.jsonl").read_text().splitlines()]
            self.assertTrue(all(set(row) == {"task_id", "prompt"} for row in public))
            self.assertEqual(set(report["test_imports_static"]), set(report["selected_ids"]))
            with self.assertRaises(FileExistsError):
                reserve(root / "parent", root / "followup")

    def test_parent_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root);self.parent(root / "parent")
            (root / "parent/dataset.parquet").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "drift"):
                reserve(root / "parent", root / "followup")
            self.assertFalse((root / "followup").exists())

    def test_parent_selection_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root);self.parent(root / "parent")
            path = root / "parent/manifest.json";packet = json.loads(path.read_text())
            packet["selected_ids"].reverse();path.write_text(json.dumps(packet))
            with self.assertRaisesRegex(ValueError, "selection mismatch"):
                reserve(root / "parent", root / "followup")


if __name__ == "__main__":
    unittest.main()
