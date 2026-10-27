"""Frozen paired revision scorer; NO test/score trace is sent to a model."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

from bigcodebench_x_secure_audit import evaluate_secure_julia
from score_bigcodebench_x import score as baseline_score, digest


def score_revisions(directory,snapshot,environment,baseline_directory):
    source=Path(__file__).parent;path=directory/"conditions.json";p=json.loads(path.read_text())
    for name,expected in p["source_sha256"].items():
        if Path(name).name!=name or digest(source/name)!=expected:
            raise ValueError("Revision source drift: "+name)
    policy_path=source/"bigcodebench_x_revision_policy.json"
    if digest(policy_path)!=p["policy_sha256"]:
        raise ValueError("Policy drift")
    policy=json.loads(policy_path.read_text())
    if [(j["id"],j["task"],j["parent_id"],j["arm"]) for j in p["jobs"]]!=[(j["id"],j["task"],j["parent_id"],j["mode"]) for j in policy["order"]]:
        raise ValueError("Paired cohort/order drift")
    if digest(baseline_directory/"conditions.json")!=p["baseline_conditions_sha256"]:
        raise ValueError("Baseline conditions drift")
    for name,expected in p["baseline_generation_sha256"].items():
        if Path(name).name!=name or digest(baseline_directory/(name+".json"))!=expected:
            raise ValueError("Baseline output drift")
    original=baseline_score(baseline_directory,snapshot,environment)
    if not original["complete"]:
        raise ValueError("Baseline incomplete")
    q=json.loads((source/"bigcodebench_x_secure_audit_validation.json").read_text())
    if not q["complete"] or q["tests"]!=4:
        raise ValueError("Test-blind runtime unqualified")
    for name,expected in q["source_sha256"].items():
        if digest(source/name)!=expected:
            raise ValueError("Runtime qualification drift")
    original_rows={r["id"]:r for r in original["rows"]}
    records={r["task_id"]:r for r in map(json.loads,(snapshot/"PRIVATE_selected_reference_tests.jsonl").read_text().splitlines())}
    rows=[]
    for job in p["jobs"]:
        candidate=directory/(job["id"]+".json");attempt=directory/(job["id"]+"_attempt.json")
        if not candidate.exists():
            rows.append({"id":job["id"],"status":"interrupted_attempt" if attempt.exists() else "pending"});continue
        c=json.loads(candidate.read_text())
        if not job["eligible"]:
            if c.get("not_generated") is not True or c["reason"]!=job["skip_reason"] or c["conditions_sha256"]!=digest(path):
                raise ValueError("Skip receipt drift")
            rows.append({"id":job["id"],"task":job["task"],"arm":job["arm"],"status":job["skip_reason"]});continue
        if c["attempt_sha256"]!=digest(attempt):
            raise ValueError("Attempt drift")
        if "inference_error" in c:
            rows.append({"id":job["id"],"task":job["task"],"arm":job["arm"],"status":"inference_error"});continue
        if c["conditions_sha256"]!=digest(path) or c["messages"]!=job["messages"]:
            raise ValueError("Prompt/conditions drift")
        if c["budget_tokens"]!=p["decoding"]["max_tokens"] or c["thinking_budget"]!=p["thinking_budget"]:
            raise ValueError("Budget drift")
        if c["decode_submissions"]!=c["decode_steps_executed"] or c["sampling"]["draws"]!=c["decode_steps_executed"]+1:
            raise ValueError("Submission/sampling invariant")
        for field in ("seed","topk","temperature","top_p","presence"):
            a,b=c["sampling"][field],p["decoding"][field]
            if field in ("temperature","top_p","presence"):
                a,b=struct.pack("f",a),struct.pack("f",b)
            if a!=b:
                raise ValueError("Sampling drift")
        tests=records[job["task"]]["test_suite"];output=directory/(job["id"]+"_score.json")
        expected={"program_sha256":hashlib.sha256(c["code"].encode()).hexdigest(),"test_suite_sha256":hashlib.sha256(tests.encode()).hexdigest(),"scorer_sha256":digest(Path(__file__))}
        if output.exists():
            result=json.loads(output.read_text())
            if any(result[k]!=v for k,v in expected.items()):
                raise ValueError("Cached revision score drift")
        else:
            result=evaluate_secure_julia(c["code"],tests,environment);result.update(expected)
            output.write_text(json.dumps(result,indent=2)+"\n")
        rows.append({"id":job["id"],"task":job["task"],"arm":job["arm"],"status":"passed" if result["passed"] else "failed",
                     "baseline_status":original_rows[job["parent_id"]]["status"],"truncated":c["truncated"],"generation_sha256":digest(candidate)})
    terminal={"passed","failed","inference_error","upstream_inference_error","paired_context_capacity_failure"}
    report={"scope":p["scope"],"policy":p["policy"],"rows":rows,"complete":all(r["status"] in terminal for r in rows),
            "conditions_sha256":digest(path),"scorer_sha256":digest(Path(__file__)),"runtime":"Qualified test-blind CPU-only Julia jail, unchanged original external tests"}
    (directory/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ("directory","snapshot","environment","baseline_directory"):
        parser.add_argument(name,type=Path)
    a=parser.parse_args();print(json.dumps(score_revisions(a.directory,a.snapshot,a.environment,a.baseline_directory)))
