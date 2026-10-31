import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import score_alias_transfer3 as module


class AliasTransferScoreTests(unittest.TestCase):
    def receipt(self, job, decoding):
        messages = json.dumps(job["messages"], ensure_ascii=False, separators=(",", ":")).encode()
        return {"id": job["id"], "conditions_sha256": "conditions", "attempt_sha256": "attempt",
                "messages": job["messages"], "normal_final_waitidle": True,
                "input_sha256": hashlib.sha256(messages).hexdigest(), "input_tokens": job["prompt_tokens"],
                "budget_tokens": decoding["max_tokens"], "decode_steps_executed": 64, "decode_submissions": 64,
                "eos_output_poll_chunk": 64, "truncated": False, "code": "error(\"never execute here\")",
                "sampling": {**{k: decoding[k] for k in ("seed", "topk", "temperature", "top_p", "presence")}, "draws": 65}}

    def test_receipt_fail_closed(self):
        job = {"id": "alias3_0_direct", "messages": [{"user": "Unicode α/😀\nquoted \"reference\""}], "prompt_tokens": 10, "eligible": True}
        decoding = {"max_tokens": 4096, "seed": 20261025, "topk": 20, "temperature": 0.6, "top_p": 0.95, "presence": 0, "output_chunk": 64}
        p = {"decoding": decoding}
        c = self.receipt(job, decoding)
        self.assertIsNone(module.validate_candidate(c, job, p, "conditions", "attempt"))
        for key, value in (("id", "wrong"), ("conditions_sha256", "wrong"), ("attempt_sha256", "wrong"),
                           ("messages", []), ("normal_final_waitidle", False), ("input_sha256", "wrong"),
                           ("input_tokens", 11), ("budget_tokens", 2048), ("decode_steps_executed", 4097),
                           ("decode_submissions", 63), ("eos_output_poll_chunk", 128), ("truncated", 1)):
            changed = {**c, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                module.validate_candidate(changed, job, p, "conditions", "attempt")
        changed = copy.deepcopy(c)
        changed["sampling"]["draws"] = 64
        with self.assertRaises(ValueError):
            module.validate_candidate(changed, job, p, "conditions", "attempt")
        changed = copy.deepcopy(c)
        changed["sampling"]["seed"] = 1
        with self.assertRaises(ValueError):
            module.validate_candidate(changed, job, p, "conditions", "attempt")
        self.assertEqual(module.validate_candidate({**c, "inference_error": "failed"}, job, p, "conditions", "attempt"), "inference_error")
        skipped = {"id": job["id"], "conditions_sha256": "conditions", "attempt_sha256": "attempt", "not_generated": True,
                   "reason": "paired_context_capacity_failure", "gpu_setup": False}
        self.assertEqual(module.validate_candidate(skipped, {**job, "eligible": False}, p, "conditions", "attempt"), "paired_context_capacity_failure")

    def test_all_pairs_unchanged_tests_and_immutable_cached_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, project, snapshot, directory = [root / name for name in ("source", "project", "snapshot", "results")]
            for path in (source, project, snapshot, directory):
                path.mkdir()
            for name in ("Project.toml", "Manifest.toml"):
                (project / name).write_text("# pinned fixture")
            (source / "runtime.jl").write_text("# public runtime")
            (source / "helper.py").write_text("# never execute fixture")
            (source / "bigcodebench_x_secure_audit_validation.json").write_text('{"complete":true,"tests":4}')
            ids = [f"External/{i}" for i in range(8)]
            private = [{"task_id": task, "test_suite": "PRIVATE_TEST_" + task} for task in ids]
            reference_path = snapshot / "PRIVATE_selected_reference_tests.jsonl"
            reference_path.write_text("\n".join(map(json.dumps, private)) + "\n")
            manifest = {"selected_ids": ids, "excluded_prior_ids": [f"Prior/{i}" for i in range(16)],
                        "files_sha256": {reference_path.name: module.digest(reference_path)}}
            (snapshot / "manifest.json").write_text(json.dumps(manifest))
            reference = {"complete": True, "all_reference_tests_passed": True, "rows": [{}] * 8,
                         "snapshot_manifest_sha256": module.digest(snapshot / "manifest.json")}
            (snapshot / "reference_validation.json").write_text(json.dumps(reference))
            jobs = []
            for index, task in enumerate(ids):
                for arm in (("direct", "public_docs") if index % 2 == 0 else ("public_docs", "direct")):
                    jobs.append({"id": f"alias3_{index}_{arm}", "task": task, "arm": arm, "eligible": True,
                                 "messages": [{"system": "contract"}, {"user": "Public only α " + arm}], "prompt_tokens": 10})
            decoding = {"max_tokens": 4096, "seed": 20261025, "topk": 20, "temperature": 0.6, "top_p": 0.95, "presence": 0, "output_chunk": 64}
            p = {"scope": "MOCK scoring only", "policy": "no candidates executed", "jobs": jobs, "decoding": decoding,
                 "private_scorer_sha256": module.digest(Path(module.__file__)),
                 "source_sha256": {"runtime.jl": module.digest(source / "runtime.jl")},
                 "secure_runtime_sources": {"helper.py": module.digest(source / "helper.py")},
                 "secure_runtime_validation_sha256": module.digest(source / "bigcodebench_x_secure_audit_validation.json"),
                 "snapshot_manifest_sha256": module.digest(snapshot / "manifest.json"),
                 "reference_validation_sha256": module.digest(snapshot / "reference_validation.json"),
                 "project_sha256": module.digest(project / "Project.toml"), "manifest_sha256": module.digest(project / "Manifest.toml")}
            conditions = directory / "conditions.json"
            conditions.write_text(json.dumps(p))
            for job in jobs:
                attempt = directory / (job["id"] + "_attempt.json")
                attempt.write_text('{"scope":"MOCK"}')
                c = self.receipt(job, decoding)
                c.update(conditions_sha256=module.digest(conditions), attempt_sha256=module.digest(attempt))
                (directory / (job["id"] + ".json")).write_text(json.dumps(c))
            with patch.object(module, "SOURCE", source), patch.object(module, "ROOT", project), \
                    patch.object(module, "evaluate_secure_julia", return_value={"passed": True}) as evaluate:
                result = module.score(directory, snapshot, root / "unused")
                self.assertTrue(result["complete"])
                self.assertEqual(len(result["rows"]), 16)
                self.assertEqual(evaluate.call_count, 16)
                self.assertEqual([call.args[1] for call in evaluate.call_args_list], ["PRIVATE_TEST_" + j["task"] for j in jobs])
                evaluate.reset_mock()
                module.score(directory, snapshot, root / "unused")
                evaluate.assert_not_called()
                changed = directory / (jobs[0]["id"] + "_score.json")
                payload = json.loads(changed.read_text())
                payload["program_sha256"] = "changed"
                changed.write_text(json.dumps(payload))
                with self.assertRaises(ValueError):
                    module.score(directory, snapshot, root / "unused")


if __name__ == "__main__":
    unittest.main()
