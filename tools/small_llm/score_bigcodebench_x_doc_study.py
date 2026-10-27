"""Prospective paired public-document retrieval, test-blind external scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

from bigcodebench_x_secure_audit import evaluate_secure_julia
from bigcodebench_x_julia_adapter import ROOT

SOURCE = Path(__file__).parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_order(ids):
    return [(task, arm) for i, task in enumerate(ids) for arm in
            (("direct", "public_docs") if i % 2 == 0 else ("public_docs", "direct"))]


def score(directory, snapshot, environment):
    path = directory / "conditions.json"
    p = json.loads(path.read_text())
    for filename, expected in p["source_sha256"].items():
        if Path(filename).name != filename or digest(SOURCE / filename) != expected:
            raise ValueError("Source drift: " + filename)
    for filename, key in (("manifest.json", "snapshot_manifest_sha256"),
                          ("reference_validation.json", "reference_validation_sha256")):
        if digest(snapshot / filename) != p[key]:
            raise ValueError("Snapshot receipt drift")
    manifest = json.loads((snapshot / "manifest.json").read_text())
    reference = json.loads((snapshot / "reference_validation.json").read_text())
    if not reference["complete"] or not reference["all_reference_tests_passed"] or len(reference["rows"]) != 8:
        raise ValueError("Unqualified external references")
    if reference["snapshot_manifest_sha256"] != digest(snapshot / "manifest.json"):
        raise ValueError("Reference provenance drift")
    for filename, expected in manifest["files_sha256"].items():
        if Path(filename).name != filename or digest(snapshot / filename) != expected:
            raise ValueError("External input drift")
    if len(manifest["selected_ids"]) != 8 or set(manifest["selected_ids"]) & set(manifest["excluded_prior_ids"]):
        raise ValueError("Prior task contamination")
    if [(job["task"], job["arm"]) for job in p["jobs"]] != paired_order(manifest["selected_ids"]):
        raise ValueError("Paired cohort/order drift")
    for i in range(0, len(p["jobs"]), 2):
        if p["jobs"][i]["eligible"] != p["jobs"][i + 1]["eligible"]:
            raise ValueError("Unpaired capacity selection")
    for filename, key in (("Project.toml", "project_sha256"), ("Manifest.toml", "manifest_sha256")):
        if digest(ROOT / filename) != p[key]:
            raise ValueError("Julia dependency drift")
    qualification = json.loads((SOURCE / "bigcodebench_x_secure_audit_validation.json").read_text())
    if not qualification["complete"] or qualification["tests"] != 4:
        raise ValueError("Unqualified test-blind runtime")
    for filename, expected in qualification["source_sha256"].items():
        if Path(filename).name != filename or digest(SOURCE / filename) != expected:
            raise ValueError("Runtime source drift")
    records = {row["task_id"]: row for row in map(json.loads, (snapshot / "PRIVATE_selected_reference_tests.jsonl").read_text().splitlines())}
    if list(records) != manifest["selected_ids"]:
        raise ValueError("Private/public cohort drift")
    rows = []
    for job in p["jobs"]:
        expected_id = "doc_" + job["task"].rsplit("/", 1)[-1] + "_" + job["arm"]
        if job["id"] != expected_id or Path(job["id"]).name != job["id"]:
            raise ValueError("Unsafe candidate id")
        candidate = directory / (job["id"] + ".json")
        attempt = directory / (job["id"] + "_attempt.json")
        if not candidate.exists():
            rows.append({"id": job["id"], "status": "interrupted_attempt" if attempt.exists() else "pending"})
            continue
        c = json.loads(candidate.read_text())
        if not job["eligible"]:
            if c.get("not_generated") is not True or c.get("reason") != "paired_context_capacity_failure" or c["conditions_sha256"] != digest(path):
                raise ValueError("Skip receipt drift")
            rows.append({"id": job["id"], "task": job["task"], "arm": job["arm"], "status": "paired_context_capacity_failure"})
            continue
        if c["attempt_sha256"] != digest(attempt):
            raise ValueError("Attempt drift")
        if "inference_error" in c:
            rows.append({"id": job["id"], "task": job["task"], "arm": job["arm"], "status": "inference_error"})
            continue
        if c["conditions_sha256"] != digest(path) or c["messages"] != job["messages"]:
            raise ValueError("Prompt/conditions drift")
        if c["budget_tokens"] != p["decoding"]["max_tokens"] or c["thinking_budget"] != p["thinking_budget"]:
            raise ValueError("Budget drift")
        if c["decode_submissions"] != c["decode_steps_executed"] or c["sampling"]["draws"] != c["decode_steps_executed"] + 1:
            raise ValueError("Submission/sampling invariant")
        for field in ("seed", "topk", "temperature", "top_p", "presence"):
            actual, expected = c["sampling"][field], p["decoding"][field]
            if field in ("temperature", "top_p", "presence"):
                actual, expected = struct.pack("f", actual), struct.pack("f", expected)
            if actual != expected:
                raise ValueError("Sampling drift")
        tests = records[job["task"]]["test_suite"]
        hashes = {"program_sha256": hashlib.sha256(c["code"].encode()).hexdigest(),
                  "test_suite_sha256": hashlib.sha256(tests.encode()).hexdigest(),
                  "scorer_sha256": digest(Path(__file__))}
        scorepath = directory / (job["id"] + "_score.json")
        if scorepath.exists():
            result = json.loads(scorepath.read_text())
            if any(result[key] != value for key, value in hashes.items()):
                raise ValueError("Cached score drift")
        else:
            result = evaluate_secure_julia(c["code"], tests, environment)
            result.update(hashes)
            scorepath.write_text(json.dumps(result, indent=2) + "\n")
        rows.append({"id": job["id"], "task": job["task"], "arm": job["arm"],
                     "status": "passed" if result["passed"] else "failed",
                     "truncated": c["truncated"], "generation_sha256": digest(candidate)})
    report = {"scope": p["scope"], "policy": p["policy"], "rows": rows,
              "complete": all(row["status"] in ("passed", "failed", "inference_error", "paired_context_capacity_failure") for row in rows),
              "conditions_sha256": digest(path), "scorer_sha256": digest(Path(__file__)),
              "runtime": "Qualified test-blind CPU-only jail; unchanged external tests"}
    (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("directory", "snapshot", "environment"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(json.dumps(score(args.directory, args.snapshot, args.environment)))
