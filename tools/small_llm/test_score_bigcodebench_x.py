"""Private receipt/cache fixtures. Mocked execution is NOT model-quality data."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import score_bigcodebench_x as grader

CONDITIONS=Path(sys.argv.pop(1)).read_text()
SNAPSHOT=Path(sys.argv.pop(1))
ENVIRONMENT=Path(sys.executable).absolute().parent.parent


class ScorerTests(unittest.TestCase):
    def fixture(self, directory):
        (directory/"conditions.json").write_text(CONDITIONS)
        p=json.loads(CONDITIONS)
        for job in p["jobs"]:
            attempt=directory/(job["id"]+"_attempt.json")
            attempt.write_text('{"private_fixture":true}')
            row={"attempt_sha256":grader.digest(attempt),"conditions_sha256":grader.digest(directory/"conditions.json"),
                 "messages":job["messages"],"budget_tokens":6144,"thinking_budget":4096,
                 "decode_submissions":4,"decode_steps_executed":4,"sampling":{**p["decoding"],"draws":5},
                 "code":"println(17)","truncated":False,"generated_tokens":4}
            (directory/(job["id"]+".json")).write_text(json.dumps(row))
        return p

    def test_all_receipts_and_cache(self):
        with tempfile.TemporaryDirectory() as root:
            directory=Path(root);p=self.fixture(directory)
            def trusted_mock(program, tests, environment):
                return {"passed":True,"program_sha256":hashlib.sha256(program.encode()).hexdigest(),
                        "test_suite_sha256":hashlib.sha256(tests.encode()).hexdigest()}
            with patch.object(grader,"evaluate_julia",side_effect=trusted_mock) as execute:
                report=grader.score(directory,SNAPSHOT,ENVIRONMENT)
                self.assertTrue(report["complete"])
                self.assertEqual(execute.call_count,8)
                self.assertTrue(all(r["status"]=="passed" for r in report["rows"]))
            with patch.object(grader,"evaluate_julia",side_effect=RuntimeError("must use cache")):
                self.assertEqual(grader.score(directory,SNAPSHOT,ENVIRONMENT)["rows"],report["rows"])
            candidate=directory/(p["jobs"][0]["id"]+".json")
            bad=json.loads(candidate.read_text());bad["decode_submissions"]=5;candidate.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError,"submission"):
                grader.score(directory,SNAPSHOT,ENVIRONMENT)

    def test_missing_outputs_stay_pending(self):
        with tempfile.TemporaryDirectory() as root:
            directory=Path(root);(directory/"conditions.json").write_text(CONDITIONS)
            report=grader.score(directory,SNAPSHOT,ENVIRONMENT)
            self.assertFalse(report["complete"])
            self.assertTrue(all(r["status"]=="pending" for r in report["rows"]))


if __name__=="__main__":
    result=unittest.main(exit=False).result
    directory=Path(__file__).parent
    report={"complete":result.wasSuccessful(),"tests":result.testsRun,
            "scope":"PRIVATE mocked execution/receipt/cache qualification; not model-quality evidence",
            "source_sha256":{name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in
                ("score_bigcodebench_x.py","test_score_bigcodebench_x.py")}}
    (directory/"bigcodebench_x_scorer_validation.json").write_text(json.dumps(report,indent=2)+"\n")
    sys.exit(0 if result.wasSuccessful() else 1)
