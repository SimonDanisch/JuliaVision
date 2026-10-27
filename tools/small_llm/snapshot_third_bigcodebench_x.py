"""Reserve positions16:24 of the original fixed ordering, before model output.

No dependency/difficulty/score filtering. Both prior cohorts are excluded and
verified against the same pinned dataset. This is public benchmark material,
not a contamination-free holdout or a completed Julia/model qualification.
"""
import argparse
import ast
import json
from pathlib import Path

import pyarrow
import pyarrow.parquet as pq
from snapshot_bigcodebench_x import digest, SELECTION_SEED


def reserve(initial, followup, destination):
    if destination.exists():
        raise FileExistsError("Preserve third snapshot")
    parents = []
    for root in (initial, followup):
        manifest = json.loads((root / "manifest.json").read_text())
        for name, expected in manifest["files_sha256"].items():
            if Path(name).name != name or digest((root / name).read_bytes()) != expected:
                raise ValueError("Parent integrity mismatch")
        if manifest["selection_seed"] != SELECTION_SEED or manifest["selected_count"] != 8:
            raise ValueError("Parent selection mismatch")
        parents.append(manifest)
    first, second = parents
    if first["files_sha256"]["dataset.parquet"] != second["files_sha256"]["dataset.parquet"]:
        raise ValueError("Different dataset bytes")
    if second["parent_manifest_sha256"] != digest((initial / "manifest.json").read_bytes()):
        raise ValueError("Parent lineage mismatch")
    raw = (initial / "dataset.parquet").read_bytes()
    rows = pq.read_table(pyarrow.BufferReader(raw)).to_pylist()
    ids = [r["task_id"] for r in rows]
    if len(ids) != len(set(ids)) or len(ids) != first["total_records"]:
        raise ValueError("Duplicate/inconsistent dataset")
    ordered = sorted(rows, key=lambda r: digest((SELECTION_SEED + "\0" + r["task_id"]).encode()))
    if [r["task_id"] for r in ordered[:8]] != first["selected_ids"] or [r["task_id"] for r in ordered[8:16]] != second["selected_ids"]:
        raise ValueError("Prior ordering mismatch")
    selected = ordered[16:24]
    if len(selected) != 8:
        raise ValueError("Incomplete third cohort")
    imports = {}
    for row in selected:
        modules = set()
        for node in ast.walk(ast.parse(row["test_suite"])):
            if isinstance(node, ast.Import):
                modules.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module.split(".")[0])
        imports[row["task_id"]] = sorted(modules)
    private = b"".join((json.dumps(r, ensure_ascii=False) + "\n").encode() for r in selected)
    public = b"".join((json.dumps({k: r[k] for k in ("task_id", "prompt")}, ensure_ascii=False) + "\n").encode() for r in selected)
    packet = {
        "complete": True, "scope": "Third fixed-hash external cohort reservation ONLY; no generated candidates/quality scores. Public contamination possible.",
        **{k: first[k] for k in ("dataset", "revision", "repository_commit", "reference_language")},
        "selection_policy": "Positions16:24 zero-based in original SHA256(seed+NUL+task_id) ordering. No dependency, difficulty, observed score or availability filtering; retain all selected tasks.",
        "selection_seed": SELECTION_SEED, "selected_count": 8,
        "selected_ids": [r["task_id"] for r in selected],
        "excluded_prior_ids": first["selected_ids"] + second["selected_ids"],
        "total_records": len(rows), "test_imports_static": imports,
        "parent_manifests_sha256": [digest((root / "manifest.json").read_bytes()) for root in (initial, followup)],
        "snapshotter_sha256": digest(Path(__file__).read_bytes()),
        "files_sha256": {"dataset.parquet": digest(raw), "PRIVATE_selected_reference_tests.jsonl": digest(private), "public_prompts.jsonl": digest(public)},
        "not_yet_qualified": "Original reference self-tests, local Julia adapter, retrieval context and final generation/grading policy remain unqualified/unfrozen; do not generate yet.",
    }
    destination.mkdir(parents=True)
    for name, data in (("dataset.parquet", raw), ("PRIVATE_selected_reference_tests.jsonl", private), ("public_prompts.jsonl", public)):
        (destination / name).write_bytes(data)
    (destination / "manifest.json").write_text(json.dumps(packet, indent=2) + "\n")
    return packet


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("initial", type=Path)
    parser.add_argument("followup", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    result = reserve(args.initial, args.followup, args.destination)
    print(json.dumps({"selected_ids": result["selected_ids"], "test_imports_static": result["test_imports_static"]}))
