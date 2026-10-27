"""Pin external evaluation data; NEVER execute reference/test/generated code.

Run with the isolated gen/bcb-x-tools environment (pyarrow20.0.0). Public
prompts are exported separately from Python reference programs and test suites.
The fixed hash selection does not filter by task difficulty or model outcomes.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import pyarrow
import pyarrow.parquet as pq

DATASET = "nuprl-staging/BigCodeBench-MultiPL"
REVISION = "62fe7c92d64ee34d75332d06501b1e828e8f84a0"
REPOSITORY_COMMIT = "c0a4d325b185946bb85254c7978f3801b342f094"
SELECTION_SEED = "bcbx-julia-20261025"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def snapshot(destination, count=8):
    if destination.exists():
        raise FileExistsError("Preserve existing snapshot")
    if not 1 <= count <= 100:
        raise ValueError("Bounded selection required")
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/data/test-00000-of-00001.parquet"
    raw = urlopen(url, timeout=120).read(32 * 1024 * 1024 + 1)
    if len(raw) > 32 * 1024 * 1024:
        raise ValueError("Oversize snapshot")
    table = pq.read_table(pyarrow.BufferReader(raw))
    required = {"task_id", "prompt", "program", "test_suite"}
    if not required.issubset(table.column_names):
        raise ValueError(f"Unexpected schema: {table.column_names}")
    records = table.to_pylist()
    ids = [r["task_id"] for r in records]
    if len(ids) != len(set(ids)) or not records:
        raise ValueError("Duplicate/empty dataset")
    for row in records:
        if not all(isinstance(row[k], str) and row[k] for k in required):
            raise ValueError("Invalid record")
    ordered = sorted(records, key=lambda r: digest((SELECTION_SEED + "\0" + r["task_id"]).encode()))
    selected = ordered[:count]
    # Dependency inspection is static only. Never import or run test code here.
    dependencies = {}
    for row in selected:
        tree = ast.parse(row["test_suite"])
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module.split(".")[0])
        dependencies[row["task_id"]] = sorted(modules)
    destination.mkdir(parents=True)
    (destination / "dataset.parquet").write_bytes(raw)
    private = b"".join((json.dumps(r, ensure_ascii=False) + "\n").encode() for r in selected)
    public = b"".join((json.dumps({k: r[k] for k in ("task_id", "prompt")}, ensure_ascii=False) + "\n").encode() for r in selected)
    (destination / "PRIVATE_selected_reference_tests.jsonl").write_bytes(private)
    (destination / "public_prompts.jsonl").write_bytes(public)
    manifest = {
        "complete": True,
        "scope": "External public benchmark preparation ONLY; no model generation or score. Public benchmark contamination possible. No claim of Makie/GPU or idiomatic-code coverage.",
        "dataset": DATASET, "revision": REVISION, "download_url": url,
        "primary_repository": "https://github.com/arjunguha/BigCodeBench-X",
        "repository_commit": REPOSITORY_COMMIT,
        "reference_language": "Python; original tests interact with candidate programs via stdin/stdout",
        "selection_policy": "First8 by fixed SHA256(seed+NUL+task_id), no score/availability filtering; every selected task retained including unsupported dependencies",
        "selection_seed": SELECTION_SEED, "selected_ids": [r["task_id"] for r in selected],
        "total_records": len(records), "selected_count": count,
        "test_imports_static": dependencies, "pyarrow_version": pyarrow.__version__,
        "files_sha256": {"dataset.parquet": digest(raw), "PRIVATE_selected_reference_tests.jsonl": digest(private), "public_prompts.jsonl": digest(public)},
        "snapshotter_sha256": digest(Path(__file__).read_bytes()),
        "not_yet_qualified": "Need sandboxed original-test harness/reference self-check, installed Julia dependency contract, and frozen generation protocol before any candidate generation.",
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    result = snapshot(args.destination)
    print(json.dumps({k: result[k] for k in ("total_records", "selected_count", "selected_ids")}))
