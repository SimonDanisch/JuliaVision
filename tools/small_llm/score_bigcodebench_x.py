"""Score immutable Laptop candidates with UNCHANGED pinned external tests."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

from bigcodebench_x_julia_adapter import evaluate_julia, ROOT


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(directory, snapshot, environment):
    path = directory / "conditions.json"
    p = json.loads(path.read_text())
    sources = Path(__file__).parent
    for name, expected in p["source_sha256"].items():
        if Path(name).name != name or digest(sources / name) != expected:
            raise ValueError("Source drift: " + name)
    for filename, key in (("manifest.json", "snapshot_manifest_sha256"), ("reference_validation_v2.json", "reference_validation_sha256")):
        if digest(snapshot / filename) != p[key]:
            raise ValueError("External receipt drift")
    manifest = json.loads((snapshot / "manifest.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        if Path(name).name != name or digest(snapshot / name) != expected:
            raise ValueError("External input drift")
    for filename, key in (("Project.toml", "project_sha256"), ("Manifest.toml", "manifest_sha256")):
        if digest(ROOT / filename) != p[key]:
            raise ValueError("Julia dependency drift")
    if digest(sources / "bigcodebench_x_julia_adapter_validation.json") != p["adapter_validation_sha256"]:
        raise ValueError("Adapter qualification drift")
    records = {r["task_id"]: r for r in map(json.loads, (snapshot / "PRIVATE_selected_reference_tests.jsonl").read_text().splitlines())}
    if [j["task"] for j in p["jobs"]] != manifest["selected_ids"]:
        raise ValueError("Cohort drift")
    rows = []
    for job in p["jobs"]:
        if Path(job["id"]).name != job["id"]:
            raise ValueError("Unsafe candidate id")
        candidate = directory / (job["id"] + ".json")
        attempt = directory / (job["id"] + "_attempt.json")
        if not candidate.exists():
            rows.append({"id": job["id"], "status": "interrupted_attempt" if attempt.exists() else "pending"})
            continue
        c = json.loads(candidate.read_text())
        if c["attempt_sha256"] != digest(attempt):
            raise ValueError("Attempt drift")
        if "inference_error" in c:
            rows.append({"id": job["id"], "status": "inference_error"})
            continue
        if c["conditions_sha256"] != digest(path) or c["messages"] != job["messages"]:
            raise ValueError("Prompt/conditions drift")
        if c["budget_tokens"] != p["decoding"]["max_tokens"] or c["thinking_budget"] != p["thinking_budget"]:
            raise ValueError("Budget drift")
        if c["decode_submissions"] != c["decode_steps_executed"] or c["sampling"]["draws"] != c["decode_steps_executed"] + 1:
            raise ValueError("GPU submission/sampling invariant")
        for field in ("seed", "topk", "temperature", "top_p", "presence"):
            a, b = c["sampling"][field], p["decoding"][field]
            if field in ("temperature", "top_p", "presence"):
                a, b = struct.pack("f", a), struct.pack("f", b)
            if a != b:
                raise ValueError("Sampling drift")
        code = c["code"]
        tests = records[job["task"]]["test_suite"]
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        test_hash = hashlib.sha256(tests.encode()).hexdigest()
        scorepath = directory / (job["id"] + "_score.json")
        if scorepath.exists():
            result = json.loads(scorepath.read_text())
            if (result["program_sha256"], result["test_suite_sha256"], result["scorer_sha256"]) != (code_hash, test_hash, digest(Path(__file__))):
                raise ValueError("Cached score drift")
        else:
            result = evaluate_julia(code, tests, environment)
            result["scorer_sha256"] = digest(Path(__file__))
            scorepath.write_text(json.dumps(result, indent=2) + "\n")
        rows.append({"id": job["id"], "task": job["task"], "status": "passed" if result["passed"] else "failed",
                     "truncated": c["truncated"], "generated_tokens": c["generated_tokens"], "generation_sha256": digest(candidate)})
    report = {"scope": p["scope"], "policy": p["policy"], "rows": rows,
              "complete": all(r["status"] in ("passed", "failed", "inference_error") for r in rows),
              "conditions_sha256": digest(path), "scorer_sha256": digest(Path(__file__))}
    (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("environment", type=Path)
    args = parser.parse_args()
    print(json.dumps(score(args.directory, args.snapshot, args.environment)))
