"""Uniform extra isolation audit over ALL same frozen baseline outputs."""
import argparse
import hashlib
import json
from pathlib import Path

from bigcodebench_x_secure_audit import evaluate_secure_julia
from score_bigcodebench_x import score, digest


def audit(directory, snapshot, environment, destination):
    sources=Path(__file__).parent
    qualification=sources/"bigcodebench_x_secure_audit_validation.json"
    q=json.loads(qualification.read_text())
    if not q["complete"] or q["tests"]!=4:
        raise ValueError("Test-blind execution not qualified")
    for name,expected in q["source_sha256"].items():
        if Path(name).name!=name or digest(sources/name)!=expected:
            raise ValueError("Audit qualification drift")
    original=score(directory,snapshot,environment) # validates original receipts; cached scores retained
    p=json.loads((directory/"conditions.json").read_text())
    records={r["task_id"]:r for r in map(json.loads,(snapshot/"PRIVATE_selected_reference_tests.jsonl").read_text().splitlines())}
    registration={"scope":"POSTHOC mandatory test-blind isolation audit of ALL unchanged external baseline outputs; not new neural attempts or replacement scores. Changed outcomes can reflect isolation/runtime/randomness, not automatically cheating.",
                  "parent_conditions_sha256":digest(directory/"conditions.json"),
                  "snapshot_manifest_sha256":digest(snapshot/"manifest.json"),
                  "qualification_sha256":digest(qualification),"ids":[j["id"] for j in p["jobs"]],
                  "source_sha256":{name:digest(sources/name) for name in
                       ("run_bigcodebench_x_secure_audit.py","bigcodebench_x_secure_audit.py","score_bigcodebench_x.py","bigcodebench_x_julia_adapter.py","qualify_bigcodebench_x_reference.py")}}
    destination.mkdir(parents=True,exist_ok=True)
    path=destination/"conditions.json"
    if path.exists():
        if json.loads(path.read_text())!=registration:
            raise ValueError("Preserve audit registration")
    else:
        path.write_text(json.dumps(registration,indent=2)+"\n")
    rows=[]
    old_rows={r["id"]:r for r in original["rows"]}
    for job in p["jobs"]:
        prior=old_rows[job["id"]]
        if prior["status"] not in ("passed","failed"):
            rows.append({"id":job["id"],"status":prior["status"]});continue
        candidate=directory/(job["id"]+".json");c=json.loads(candidate.read_text())
        if digest(candidate)!=prior["generation_sha256"]:
            raise ValueError("Candidate drift")
        tests=records[job["task"]]["test_suite"];output=destination/(job["id"]+"_audit.json")
        expected={"generation_sha256":digest(candidate),"original_score_sha256":digest(directory/(job["id"]+"_score.json")),
                  "audit_conditions_sha256":digest(path),"program_sha256":hashlib.sha256(c["code"].encode()).hexdigest(),
                  "test_suite_sha256":hashlib.sha256(tests.encode()).hexdigest()}
        if output.exists():
            result=json.loads(output.read_text())
            if any(result[k]!=v for k,v in expected.items()):
                raise ValueError("Cached audit drift")
        else:
            result=evaluate_secure_julia(c["code"],tests,environment)
            result.update(expected)
            output.write_text(json.dumps(result,indent=2)+"\n")
        rows.append({"id":job["id"],"task":job["task"],"status":"passed" if result["passed"] else "failed",
                     "original_status":prior["status"],"same_outcome":result["passed"]==(prior["status"]=="passed")})
    result={"scope":registration["scope"],"conditions_sha256":digest(path),"rows":rows,
            "complete":all(r["status"] in ("passed","failed","inference_error") for r in rows)}
    (destination/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("directory","snapshot","environment","destination"):
        p.add_argument(name,type=Path)
    a=p.parse_args();audit(a.directory,a.snapshot,a.environment,a.destination)
