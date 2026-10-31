import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq
import snapshot_third_bigcodebench_x as module


class ThirdSnapshotTests(unittest.TestCase):
    def parents(self, root, duplicate=False):
        rows = [{"task_id": f"External/{i}", "prompt": "Public request only",
                 "program": "raise RuntimeError('never execute reference')",
                 "test_suite": "import nonexistent_dependency\ndef test_cases(run):\n    raise RuntimeError('never execute tests')"}
                for i in range(32)]
        if duplicate:
            rows[-1]["task_id"] = rows[0]["task_id"]
        sink = pa.BufferOutputStream()
        pq.write_table(pa.Table.from_pylist(rows), sink)
        raw = sink.getvalue().to_pybytes()
        ordered = sorted(rows, key=lambda r: module.digest((module.SELECTION_SEED + "\0" + r["task_id"]).encode()))
        paths = [root / "initial", root / "followup"]
        for index, path in enumerate(paths):
            path.mkdir()
            (path / "dataset.parquet").write_bytes(raw)
            packet = {"selection_seed": module.SELECTION_SEED, "selected_count": 8,
                      "selected_ids": [r["task_id"] for r in ordered[index * 8:(index + 1) * 8]],
                      "total_records": len(rows), "dataset": "fixture", "revision": "pinned-fixture",
                      "repository_commit": "fixture", "reference_language": "python",
                      "files_sha256": {"dataset.parquet": module.digest(raw)}}
            if index:
                packet["parent_manifest_sha256"] = module.digest((paths[0] / "manifest.json").read_bytes())
            (path / "manifest.json").write_text(json.dumps(packet))
        return paths, ordered

    def test_next_eight_public_private_split_no_dependency_filter(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parents, ordered = self.parents(root)
            destination = root / "third"
            result = module.reserve(*parents, destination)
            self.assertEqual(result["selected_ids"], [r["task_id"] for r in ordered[16:24]])
            self.assertFalse(set(result["selected_ids"]) & set(result["excluded_prior_ids"]))
            public = [json.loads(line) for line in (destination / "public_prompts.jsonl").read_text().splitlines()]
            self.assertTrue(all(set(r) == {"task_id", "prompt"} for r in public))
            self.assertTrue(all(names == ["nonexistent_dependency"] for names in result["test_imports_static"].values()))
            with self.assertRaises(FileExistsError):
                module.reserve(*parents, destination)

    def test_parent_drift_fails_before_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parents, _ = self.parents(root)
            (parents[1] / "dataset.parquet").write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                module.reserve(*parents, root / "third")
            self.assertFalse((root / "third").exists())

    def test_wrong_lineage_fails_before_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parents, _ = self.parents(root)
            path = parents[1] / "manifest.json"
            packet = json.loads(path.read_text())
            packet["parent_manifest_sha256"] = "wrong"
            path.write_text(json.dumps(packet))
            with self.assertRaises(ValueError):
                module.reserve(*parents, root / "third")
            self.assertFalse((root / "third").exists())

    def test_duplicate_ids_fail_before_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parents, _ = self.parents(root, duplicate=True)
            with self.assertRaises(ValueError):
                module.reserve(*parents, root / "third")
            self.assertFalse((root / "third").exists())


if __name__ == "__main__":
    unittest.main()
