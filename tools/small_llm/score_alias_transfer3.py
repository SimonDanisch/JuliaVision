"""ROOT-only paired alias-aware retrieval scoring, unchanged external tests.

No candidate executes here: qualified evaluate_secure_julia uses a test-blind,
CPU-only, networkless jail. Private test/reference data never enters inference.
"""
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


def validate_candidate(c, job, p, conditions_sha256, attempt_sha256):
    if c["id"] != job["id"] or c["conditions_sha256"] != conditions_sha256 or c["attempt_sha256"] != attempt_sha256:
        raise ValueError("Receipt provenance mismatch")
    if not job["eligible"]:
        if c.get("not_generated") is not True or c.get("reason") != "paired_context_capacity_failure" or c.get("gpu_setup") is not False:
            raise ValueError("Paired capacity skip mismatch")
        return "paired_context_capacity_failure"
    if c["messages"] != job["messages"] or c.get("normal_final_waitidle") is not True:
        raise ValueError("Prompt/idle mismatch")
    if "inference_error" in c:
        return "inference_error"
    # Julia JSON3 hashes compact JSON bytes of the exact system/user messages.
    encoded = json.dumps(job["messages"], ensure_ascii=False, separators=(",", ":")).encode()
    if c["input_sha256"] != hashlib.sha256(encoded).hexdigest() or c["input_tokens"] != job["prompt_tokens"]:
        raise ValueError("Actual input mismatch")
    if c["budget_tokens"] != p["decoding"]["max_tokens"] or not 0 < c["decode_steps_executed"] <= c["budget_tokens"]:
        raise ValueError("Token budget mismatch")
    if c["decode_submissions"] != c["decode_steps_executed"] or c["sampling"]["draws"] != c["decode_steps_executed"] + 1:
        raise ValueError("Submission/draw mismatch")
    if c["eos_output_poll_chunk"] != p["decoding"]["output_chunk"] or type(c["truncated"]) is not bool:
        raise ValueError("Completion metadata mismatch")
    for field in ("seed", "topk", "temperature", "top_p", "presence"):
        a, b = c["sampling"][field], p["decoding"][field]
        if field in ("temperature", "top_p", "presence"):
            a, b = struct.pack("f", a), struct.pack("f", b)
        if a != b:
            raise ValueError("Sampling mismatch")
    return None


def score(directory, snapshot, environment):
    conditions = directory / "conditions.json"
    p = json.loads(conditions.read_text())
    if digest(Path(__file__)) != p["private_scorer_sha256"]:
        raise ValueError("Scorer drift")
    for filename, expected in p["source_sha256"].items():
        if Path(filename).name != filename or digest(SOURCE / filename) != expected:
            raise ValueError("Runtime source drift")
    for filename, expected in p["secure_runtime_sources"].items():
        if Path(filename).name != filename or digest(SOURCE / filename) != expected:
            raise ValueError("Test-blind runtime source drift")
    if digest(SOURCE / "bigcodebench_x_secure_audit_validation.json") != p["secure_runtime_validation_sha256"]:
        raise ValueError("Test-blind runtime proof drift")
    for filename, key in (("manifest.json", "snapshot_manifest_sha256"), ("reference_validation.json", "reference_validation_sha256")):
        if digest(snapshot / filename) != p[key]:
            raise ValueError("Snapshot receipt drift")
    manifest = json.loads((snapshot / "manifest.json").read_text())
    reference = json.loads((snapshot / "reference_validation.json").read_text())
    if not reference["complete"] or not reference["all_reference_tests_passed"] or len(reference["rows"]) != 8 or reference["snapshot_manifest_sha256"] != digest(snapshot / "manifest.json"):
        raise ValueError("Reference qualification incomplete")
    for filename, expected in manifest["files_sha256"].items():
        if Path(filename).name != filename or digest(snapshot / filename) != expected:
            raise ValueError("External data drift")
    if len(manifest["selected_ids"]) != 8 or len(manifest["excluded_prior_ids"]) != 16 or set(manifest["selected_ids"]) & set(manifest["excluded_prior_ids"]):
        raise ValueError("Cohort overlap")
    expected_order = [(task, arm) for i, task in enumerate(manifest["selected_ids"]) for arm in
                      (("direct", "public_docs") if i % 2 == 0 else ("public_docs", "direct"))]
    if [(j["task"], j["arm"]) for j in p["jobs"]] != expected_order:
        raise ValueError("Counterbalanced pair order drift")
    for index in range(0, 16, 2):
        if p["jobs"][index]["eligible"] != p["jobs"][index + 1]["eligible"]:
            raise ValueError("Unpaired skip")
    for filename, key in (("Project.toml", "project_sha256"), ("Manifest.toml", "manifest_sha256")):
        if digest(ROOT / filename) != p[key]:
            raise ValueError("Local Julia environment drift")
    records = {r["task_id"]: r for r in map(json.loads, (snapshot / "PRIVATE_selected_reference_tests.jsonl").read_text().splitlines())}
    if list(records) != manifest["selected_ids"]:
        raise ValueError("Private/public cohort drift")
    rows = []
    for job in p["jobs"]:
        expected_id = "alias3_" + job["task"].rsplit("/", 1)[-1] + "_" + job["arm"]
        if job["id"] != expected_id or Path(job["id"]).name != job["id"]:
            raise ValueError("Invalid job ID")
        candidate = directory / (job["id"] + ".json")
        attempt = directory / (job["id"] + "_attempt.json")
        if not candidate.exists():
            rows.append({"id": job["id"], "status": "interrupted_attempt" if attempt.exists() else "pending"})
            continue
        c = json.loads(candidate.read_text())
        failure = validate_candidate(c, job, p, digest(conditions), digest(attempt))
        if failure:
            rows.append({"id": job["id"], "task": job["task"], "arm": job["arm"], "status": failure})
            continue
        tests = records[job["task"]]["test_suite"]
        provenance = {"program_sha256": hashlib.sha256(c["code"].encode()).hexdigest(), "test_suite_sha256": hashlib.sha256(tests.encode()).hexdigest(),
                      "scorer_sha256": digest(Path(__file__)), "generation_sha256": digest(candidate)}
        scorepath = directory / (job["id"] + "_score.json")
        if scorepath.exists():
            result = json.loads(scorepath.read_text())
            if any(result.get(k) != v for k, v in provenance.items()):
                raise ValueError("Cached score drift")
        else:
            result = evaluate_secure_julia(c["code"], tests, environment)
            result.update(provenance)
            scorepath.write_text(json.dumps(result, indent=2) + "\n")
        rows.append({"id": job["id"], "task": job["task"], "arm": job["arm"], "status": "passed" if result["passed"] else "failed",
                     "truncated": c["truncated"], "generation_sha256": digest(candidate)})
    report = {"scope": p["scope"], "policy": p["policy"], "rows": rows,
              "complete": all(r["status"] in ("passed", "failed", "inference_error", "paired_context_capacity_failure") for r in rows),
              "conditions_sha256": digest(conditions), "scorer_sha256": digest(Path(__file__)),
              "runtime": "Unchanged external tests, qualified CPU-only test-blind jail; no plot-aesthetics claim"}
    (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("directory", "snapshot", "environment"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(json.dumps(score(args.directory, args.snapshot, args.environment)))
