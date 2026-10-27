"""PRIVATE protocol/cache fixtures; mocked execution is NOT quality evidence."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import score_bigcodebench_x_revisions as grader

DIRECTORY=Path(sys.argv.pop(1));BASELINE=Path(sys.argv.pop(1));SNAPSHOT=Path(sys.argv.pop(1))
ENVIRONMENT=Path(sys.executable).absolute().parent.parent


class RevisionScorerTests(unittest.TestCase):
    def test_all16_receipts_cache_and_submission_rejection(self):
        p=json.loads((DIRECTORY/"conditions.json").read_text())
        original={"complete":True,"rows":[{"id":name,"status":"failed"} for name in p["baseline_generation_sha256"]]}
        for job in p["jobs"]:
            attempt=DIRECTORY/(job["id"]+"_attempt.json");attempt.write_text('{"private":true}')
            row={"attempt_sha256":grader.digest(attempt),"conditions_sha256":grader.digest(DIRECTORY/"conditions.json"),
                 "messages":job["messages"],"budget_tokens":4096,"thinking_budget":2048,
                 "decode_submissions":4,"decode_steps_executed":4,"sampling":{**p["decoding"],"draws":5},
                 "code":"println(17)","truncated":False}
            (DIRECTORY/(job["id"]+".json")).write_text(json.dumps(row))
        def mocked(program,tests,environment):
            return {"passed":True,"program_sha256":hashlib.sha256(program.encode()).hexdigest(),"test_suite_sha256":hashlib.sha256(tests.encode()).hexdigest()}
        with patch.object(grader,"baseline_score",return_value=original),patch.object(grader,"evaluate_secure_julia",side_effect=mocked) as execute:
            result=grader.score_revisions(DIRECTORY,SNAPSHOT,ENVIRONMENT,BASELINE)
            self.assertTrue(result["complete"])
            self.assertEqual(execute.call_count,16)
            self.assertTrue(all(r["status"]=="passed" for r in result["rows"]))
        with patch.object(grader,"baseline_score",return_value=original),patch.object(grader,"evaluate_secure_julia",side_effect=RuntimeError("must use cache")):
            self.assertEqual(grader.score_revisions(DIRECTORY,SNAPSHOT,ENVIRONMENT,BASELINE)["rows"],result["rows"])
        path=DIRECTORY/(p["jobs"][0]["id"]+".json")
        bad=json.loads(path.read_text());bad["decode_submissions"]=5;path.write_text(json.dumps(bad))
        with patch.object(grader,"baseline_score",return_value=original):
            with self.assertRaisesRegex(ValueError,"Submission"):
                grader.score_revisions(DIRECTORY,SNAPSHOT,ENVIRONMENT,BASELINE)


if __name__=="__main__":
    result=unittest.main(exit=False).result
    source=Path(__file__).parent
    report={"complete":result.wasSuccessful(),"tests":result.testsRun,
            "scope":"PRIVATE mocked all16 receipt/cache checks only, NOT neural/model scores",
            "source_sha256":{f:grader.digest(source/f) for f in ("score_bigcodebench_x_revisions.py","test_score_bigcodebench_x_revisions.py","bigcodebench_x_revisions.jl")}}
    (source/"bigcodebench_x_revision_scorer_validation.json").write_text(json.dumps(report,indent=2)+"\n")
    sys.exit(0 if result.wasSuccessful() else 1)
