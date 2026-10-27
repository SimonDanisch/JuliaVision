import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

spec = importlib.util.spec_from_file_location("snapshotter", Path(__file__).with_name("snapshot_bigcodebench_x.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SnapshotTests(unittest.TestCase):
    def payload(self, duplicate=False):
        rows = [{"task_id": f"External/{i}", "prompt": "Read an integer and print it.",
                 "program": "raise RuntimeError('must never execute')",
                 "test_suite": "import json\ndef test_cases(run):\n    raise RuntimeError('must never execute')\n"}
                for i in range(10)]
        if duplicate:
            rows[-1]["task_id"] = rows[0]["task_id"]
        sink = pa.BufferOutputStream()
        pq.write_table(pa.Table.from_pylist(rows), sink)
        return sink.getvalue().to_pybytes()

    def test_static_selection_and_public_private_separation(self):
        with tempfile.TemporaryDirectory() as root:
            destination = Path(root) / "snapshot"
            with patch.object(module, "urlopen", return_value=io.BytesIO(self.payload())) as fetch:
                result = module.snapshot(destination)
            self.assertEqual(result["selected_count"], 8)
            self.assertEqual(result["total_records"], 10)
            self.assertIn(module.REVISION, fetch.call_args.args[0])
            public = [json.loads(line) for line in (destination / "public_prompts.jsonl").read_text().splitlines()]
            self.assertTrue(all(set(row) == {"task_id", "prompt"} for row in public))
            self.assertEqual([row["task_id"] for row in public], result["selected_ids"])
            self.assertTrue(all(v == ["json"] for v in result["test_imports_static"].values()))
            with self.assertRaises(FileExistsError):
                module.snapshot(destination)

    def test_duplicates_fail_before_destination_creation(self):
        with tempfile.TemporaryDirectory() as root:
            destination = Path(root) / "snapshot"
            with patch.object(module, "urlopen", return_value=io.BytesIO(self.payload(duplicate=True))):
                with self.assertRaises(ValueError):
                    module.snapshot(destination)
            self.assertFalse(destination.exists())

    def test_selection_bound(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                module.snapshot(Path(root) / "snapshot", count=0)


if __name__ == "__main__":
    unittest.main()
