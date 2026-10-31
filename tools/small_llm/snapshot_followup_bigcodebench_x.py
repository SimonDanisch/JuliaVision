"""Reserve the NEXT fixed-hash cohort; never select by observed model scores."""
import argparse
import ast
import json
from pathlib import Path

import pyarrow
import pyarrow.parquet as pq

from snapshot_bigcodebench_x import digest, SELECTION_SEED


def reserve(parent, destination):
    if destination.exists():
        raise FileExistsError("Preserve follow-up snapshot")
    manifest = json.loads((parent / "manifest.json").read_text())
    for filename, expected in manifest["files_sha256"].items():
        if Path(filename).name != filename or digest((parent / filename).read_bytes()) != expected:
            raise ValueError("Parent snapshot drift")
    if manifest["selection_seed"] != SELECTION_SEED or manifest["selected_count"] != 8:
        raise ValueError("Unexpected parent cohort")
    raw = (parent / "dataset.parquet").read_bytes()
    records = pq.read_table(pyarrow.BufferReader(raw)).to_pylist()
    ids = [row["task_id"] for row in records]
    if len(ids) != len(set(ids)) or len(ids) != manifest["total_records"]:
        raise ValueError("Dataset identity mismatch")
    ordered = sorted(records, key=lambda row: digest((SELECTION_SEED + "\0" + row["task_id"]).encode()))
    if [row["task_id"] for row in ordered[:8]] != manifest["selected_ids"]:
        raise ValueError("Parent selection mismatch")
    selected = ordered[8:16]
    if len(selected) != 8:
        raise ValueError("Insufficient next cohort")
    imports = {}
    for row in selected:
        modules = set()
        for node in ast.walk(ast.parse(row["test_suite"])):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module.split(".")[0])
        imports[row["task_id"]] = sorted(modules)
    private = b"".join((json.dumps(row, ensure_ascii=False) + "\n").encode() for row in selected)
    public = b"".join((json.dumps({key: row[key] for key in ("task_id", "prompt")}, ensure_ascii=False) + "\n").encode() for row in selected)
    packet = {
        "complete": True,
        "scope": "Prospective NEXT fixed-hash public cohort reservation only; no candidates generated or quality scored. Previously used eight excluded. Public benchmark contamination remains possible.",
        "dataset": manifest["dataset"], "revision": manifest["revision"],
        "repository_commit": manifest["repository_commit"],
        "reference_language": manifest["reference_language"],
        "selection_policy": "Positions8:16 (zero-based) in SAME SHA256(seed+NUL+task_id) ordering; no difficulty, dependency, observed score or availability filtering. Every selected task retained.",
        "selection_seed": SELECTION_SEED, "selected_count": 8,
        "selected_ids": [row["task_id"] for row in selected],
        "excluded_prior_ids": manifest["selected_ids"], "total_records": len(records),
        "parent_manifest_sha256": digest((parent / "manifest.json").read_bytes()),
        "files_sha256": {"dataset.parquet": digest(raw), "PRIVATE_selected_reference_tests.jsonl": digest(private), "public_prompts.jsonl": digest(public)},
        "test_imports_static": imports,
        "snapshotter_sha256": digest(Path(__file__).read_bytes()),
        "not_yet_qualified": "Reference self-check, Julia execution adapter and final intervention protocol must be qualified/frozen before any candidate generation. Dependency failures cannot silently remove tasks.",
    }
    destination.mkdir(parents=True)
    (destination / "dataset.parquet").write_bytes(raw)
    (destination / "PRIVATE_selected_reference_tests.jsonl").write_bytes(private)
    (destination / "public_prompts.jsonl").write_bytes(public)
    (destination / "manifest.json").write_text(json.dumps(packet, indent=2) + "\n")
    print(json.dumps({"selected_ids": packet["selected_ids"], "test_imports_static": imports}))
    return packet


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("parent", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    reserve(args.parent, args.destination)
