import json
from pathlib import Path
import tempfile
import unittest

from analyze_bcbx_failures import analyze, classify, digest


class FailureAuditTests(unittest.TestCase):
    def test_all_first_visible_categories(self):
        self.assertEqual(classify("passed", {}, {}), "passed")
        self.assertEqual(classify("pending", {}, {}), "pending")
        self.assertEqual(classify("failed", {"code": " "}, {}), "missing_final_program")
        cases = [("ParseError", "syntax_error"), ("syntax: invalid", "syntax_error"),
                 ("Package DataFrames not found", "unavailable_dependency"),
                 ("UndefVarError", "undefined_name"), ("MethodError", "method_error"),
                 ("BoundsError", "runtime_value_or_argument_error"),
                 ("ERROR: LoadError: unexpected", "other_julia_runtime_error"),
                 ("AssertionError", "test_assertion_failure"), ("unknown", "unclassified_failure")]
        for diagnostic, expected in cases:
            self.assertEqual(classify("failed", {"code": "x"}, {"stderr_tail": diagnostic}), expected)
        self.assertEqual(classify("failed", {"code": "x"}, {"returncode": -9}), "bounded_execution_failure")
        # Runtime error precedes the test wrapper's failed-exit assertion.
        self.assertEqual(classify("failed", {"code": "x"}, {"stderr_tail": "UndefVarError\nAssertionError"}), "undefined_name")

    def fixture(self, root, complete=True):
        import hashlib
        job = {"id": "a", "task": "task", "arm": "direct"}
        (root / "conditions.json").write_text(json.dumps({"jobs": [job]}))
        (root / "a.json").write_text(json.dumps({"code": "x", "truncated": False}))
        (root / "a_score.json").write_text(json.dumps({"program_sha256": hashlib.sha256(b"x").hexdigest(), "stderr_tail": "MethodError"}))
        (root / "summary.json").write_text(json.dumps({"complete": complete, "conditions_sha256": digest(root / "conditions.json"), "rows": [{"id": "a", "status": "failed", "generation_sha256": digest(root / "a.json")}]}))

    def test_complete_audit_retains_all_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);self.fixture(root)
            report = analyze(root, root / "audit.json")
            self.assertEqual(report["counts_by_arm"], {"direct": {"method_error": 1}})
            self.assertEqual(len(report["rows"]), 1)
            with self.assertRaises(FileExistsError):
                analyze(root, root / "audit.json")

    def test_incomplete_and_generation_drift_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);self.fixture(root, complete=False)
            with self.assertRaisesRegex(ValueError, "Finish"):
                analyze(root, root / "audit.json")
            self.assertFalse((root / "audit.json").exists())
            self.fixture(root);(root / "a.json").write_text('{"code":"changed"}')
            with self.assertRaisesRegex(ValueError, "Generation drift"):
                analyze(root, root / "audit.json")


if __name__ == "__main__":
    unittest.main()
