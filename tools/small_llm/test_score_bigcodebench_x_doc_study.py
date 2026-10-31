"""PRIVATE mocked protocol fixtures; not neural model accuracy evidence."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import score_bigcodebench_x_doc_study as grader


def write(path, packet):
    path.write_text(json.dumps(packet))


class DocScorerTests(unittest.TestCase):
    def fixture(self, root):
        source = root / "source";source.mkdir()
        project = root / "project";project.mkdir()
        snapshot = root / "snapshot";snapshot.mkdir()
        results = root / "results";results.mkdir()
        for filename in ("Project.toml", "Manifest.toml"):
            (project / filename).write_text("fixture")
        write(source / "bigcodebench_x_secure_audit_validation.json", {"complete": True, "tests": 4, "source_sha256": {}})
        ids = [f"External/{i}" for i in range(8)]
        private = snapshot / "PRIVATE_selected_reference_tests.jsonl"
        private.write_text("".join(json.dumps({"task_id": task, "test_suite": "PRIVATE_TEST_NOT_FOR_MODEL"}) + "\n" for task in ids))
        manifest = {"selected_ids": ids, "excluded_prior_ids": ["External/old"], "files_sha256": {private.name: grader.digest(private)}}
        write(snapshot / "manifest.json", manifest)
        write(snapshot / "reference_validation.json", {"complete": True, "all_reference_tests_passed": True, "rows": [{}] * 8, "snapshot_manifest_sha256": grader.digest(snapshot / "manifest.json")})
        jobs = [{"id": "doc_" + task.split("/")[-1] + "_" + arm, "task": task, "arm": arm, "eligible": True, "messages": [{"user": "public request"}]} for task, arm in grader.paired_order(ids)]
        decoding = {"max_tokens": 4096, "seed": 20261024, "topk": 20, "temperature": .6, "top_p": .95, "presence": 0}
        packet = {"scope": "PRIVATE mocked protocol fixture, NOT model quality", "policy": "fixture", "source_sha256": {}, "jobs": jobs, "decoding": decoding, "thinking_budget": 2048, "snapshot_manifest_sha256": grader.digest(snapshot / "manifest.json"), "reference_validation_sha256": grader.digest(snapshot / "reference_validation.json"), "project_sha256": grader.digest(project / "Project.toml"), "manifest_sha256": grader.digest(project / "Manifest.toml")}
        write(results / "conditions.json", packet)
        return source, project, snapshot, results, packet

    def candidates(self, results, packet):
        for job in packet["jobs"]:
            if not job["eligible"]:
                write(results / (job["id"] + ".json"), {"not_generated": True, "reason": "paired_context_capacity_failure", "conditions_sha256": grader.digest(results / "conditions.json")})
                continue
            attempt = results / (job["id"] + "_attempt.json");write(attempt, {"fixture": True})
            write(results / (job["id"] + ".json"), {"id": job["id"], "attempt_sha256": grader.digest(attempt), "conditions_sha256": grader.digest(results / "conditions.json"), "messages": job["messages"], "budget_tokens": 4096, "thinking_budget": 2048, "decode_submissions": 3, "decode_steps_executed": 3, "sampling": {**packet["decoding"], "draws": 4}, "code": "public fixture", "truncated": False})

    def test_all16_and_immutable_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, project, snapshot, results, packet = self.fixture(Path(temporary))
            self.candidates(results, packet)
            with patch.object(grader, "SOURCE", source), patch.object(grader, "ROOT", project), patch.object(grader, "evaluate_secure_julia", return_value={"passed": True}) as evaluator:
                report = grader.score(results, snapshot, Path("unused"))
                self.assertTrue(report["complete"]);self.assertEqual(len(report["rows"]), 16)
                self.assertEqual(evaluator.call_count, 16)
                grader.score(results, snapshot, Path("unused"));self.assertEqual(evaluator.call_count, 16)
                scorepath = results / (packet["jobs"][0]["id"] + "_score.json")
                cached = json.loads(scorepath.read_text());cached["program_sha256"] = "tampered";write(scorepath, cached)
                with self.assertRaisesRegex(ValueError, "Cached score drift"):
                    grader.score(results, snapshot, Path("unused"))

    def test_submission_and_order_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, project, snapshot, results, packet = self.fixture(Path(temporary))
            self.candidates(results, packet)
            with patch.object(grader, "SOURCE", source), patch.object(grader, "ROOT", project), patch.object(grader, "evaluate_secure_julia") as evaluator:
                output = results / (packet["jobs"][0]["id"] + ".json")
                candidate = json.loads(output.read_text());candidate["decode_submissions"] = 4;write(output, candidate)
                with self.assertRaisesRegex(ValueError, "Submission/sampling"):
                    grader.score(results, snapshot, Path("unused"))
                self.assertFalse(evaluator.called)
                packet["jobs"].reverse();write(results / "conditions.json", packet)
                with self.assertRaisesRegex(ValueError, "cohort/order"):
                    grader.score(results, snapshot, Path("unused"))

    def test_paired_skips_retained_and_unpaired_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, project, snapshot, results, packet = self.fixture(Path(temporary))
            for job in packet["jobs"][:2]:
                job["eligible"] = False
            write(results / "conditions.json", packet);self.candidates(results, packet)
            with patch.object(grader, "SOURCE", source), patch.object(grader, "ROOT", project), patch.object(grader, "evaluate_secure_julia", return_value={"passed": False}) as evaluator:
                report = grader.score(results, snapshot, Path("unused"))
                self.assertTrue(report["complete"]);self.assertEqual(evaluator.call_count, 14)
                self.assertEqual([row["status"] for row in report["rows"][:2]], ["paired_context_capacity_failure"] * 2)
                packet["jobs"][0]["eligible"] = True;write(results / "conditions.json", packet)
                with self.assertRaisesRegex(ValueError, "Unpaired capacity"):
                    grader.score(results, snapshot, Path("unused"))


if __name__ == "__main__":
    unittest.main()
