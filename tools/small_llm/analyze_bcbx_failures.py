"""ROOT-only post-execution symptom audit. NEVER a generation input.

These categories describe the first visible failure, not its unique root cause.
Passing syntax/API checks can still leave algorithmic errors undetected.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re


def classify(status, candidate, execution):
    if status == "passed":
        return "passed"
    if status != "failed":
        return status
    if not candidate.get("code", "").strip():
        return "missing_final_program"
    diagnostic = execution.get("stderr_tail", "") + "\n" + execution.get("stdout_tail", "")
    if "ParseError" in diagnostic or "syntax:" in diagnostic:
        return "syntax_error"
    if re.search(r"(?:Package .*not found|Package .*is required but does not seem to be installed)", diagnostic):
        return "unavailable_dependency"
    if "UndefVarError" in diagnostic:
        return "undefined_name"
    if "MethodError" in diagnostic:
        return "method_error"
    if any(name in diagnostic for name in ("DomainError", "DimensionMismatch", "BoundsError", "InexactError", "ArgumentError")):
        return "runtime_value_or_argument_error"
    if execution.get("returncode") in (124, 137, -9, -15, -24):
        return "bounded_execution_failure"
    if "ERROR: LoadError:" in diagnostic:
        return "other_julia_runtime_error"
    if "AssertionError" in diagnostic or "Test Failed" in diagnostic:
        return "test_assertion_failure"
    return "unclassified_failure"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(directory, output):
    if output.exists():
        raise FileExistsError("Preserve failure audit")
    summarypath = directory / "summary.json"
    summary = json.loads(summarypath.read_text())
    if not summary["complete"]:
        raise ValueError("Finish the frozen cohort before final failure analysis")
    conditions = directory / "conditions.json"
    if summary["conditions_sha256"] != digest(conditions):
        raise ValueError("Summary conditions drift")
    packet = json.loads(conditions.read_text())
    jobs = {job["id"]: job for job in packet["jobs"]}
    if [row["id"] for row in summary["rows"]] != list(jobs):
        raise ValueError("Summary cohort/order drift")
    rows = [];counts = defaultdict(Counter)
    inputs = {"conditions.json": digest(conditions), "summary.json": digest(summarypath)}
    for row in summary["rows"]:
        identifier = row["id"]
        if Path(identifier).name != identifier:
            raise ValueError("Unsafe output ID")
        candidatepath = directory / (identifier + ".json")
        scorepath = directory / (identifier + "_score.json")
        candidate = json.loads(candidatepath.read_text()) if candidatepath.exists() else {}
        execution = json.loads(scorepath.read_text()) if scorepath.exists() else {}
        if row["status"] in ("passed", "failed") and (not candidatepath.exists() or not scorepath.exists()):
            raise ValueError("Missing completed receipts")
        if "generation_sha256" in row and digest(candidatepath) != row["generation_sha256"]:
            raise ValueError("Generation drift")
        if execution and execution["program_sha256"] != hashlib.sha256(candidate["code"].encode()).hexdigest():
            raise ValueError("Program/score drift")
        for path in (candidatepath, scorepath):
            if path.exists():
                inputs[path.name] = digest(path)
        symptom = classify(row["status"], candidate, execution)
        arm = jobs[identifier]["arm"]
        counts[arm][symptom] += 1
        rows.append({"id": identifier, "task": jobs[identifier]["task"], "arm": arm,
                     "status": row["status"], "symptom": symptom,
                     "truncated": candidate.get("truncated")})
    report = {"complete": True,
              "scope": "Post-execution ROOT-only first-visible-failure taxonomy, NOT causal diagnosis, grading changes, model feedback, or proof of general intelligence. Every frozen condition retained.",
              "counts_by_arm": {arm: dict(count) for arm, count in counts.items()},
              "rows": rows, "inputs_sha256": inputs, "analyzer_sha256": digest(Path(__file__))}
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["counts_by_arm"]))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    analyze(args.directory, args.output)
