"""Pinned, fixed-hash biomedical RESEARCH-reading pilot, not clinical advice.

Official corresponding abstract context excludes the conclusion. Selection never
uses labels, topic, length or model scores. Gold/conclusions never enter public
payloads. This is an oracle-context experiment, not an evaluated retriever and
not the official train/test protocol or contamination-free clinical competence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import urllib.request

COMMIT = "1cbae8e92f72f20c8d3747cbb3bf5bc53554d997"
BASE = f"https://raw.githubusercontent.com/pubmedqa/pubmedqa/{COMMIT}/"
SALT = "small-llm-pubmedqa-reading-pilot-v1:"
LABELS = ("yes", "no", "maybe")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def partition(data, count=16):
    if type(count) is not int or not 1 <= count <= len(data):
        raise ValueError("Pilot size")
    for pmid, row in data.items():
        if not re.fullmatch(r"[0-9]+", pmid):
            raise ValueError("PMID")
        if not isinstance(row.get("QUESTION"), str) or not row["QUESTION"].strip():
            raise ValueError("Question")
        contexts = row.get("CONTEXTS")
        if not isinstance(contexts, list) or not contexts or not all(isinstance(s, str) and s.strip() for s in contexts):
            raise ValueError("Abstract context")
        if row.get("final_decision") not in LABELS:
            raise ValueError("Label")
    # Labels are validated above, but NEVER used to rank/select examples.
    ids = sorted(data, key=lambda key: sha((SALT + key).encode()))[:count]
    public = [{"pmid": pmid, "question": data[pmid]["QUESTION"], "abstract_context": data[pmid]["CONTEXTS"]} for pmid in ids]
    gold = {pmid: data[pmid]["final_decision"] for pmid in ids}
    return public, gold


def fetch(path):
    with urllib.request.urlopen(BASE + path, timeout=60) as response:
        result = response.read(16 * 1024 * 1024 + 1)
    if len(result) > 16 * 1024 * 1024:
        raise ValueError("Source size")
    return result


def snapshot(destination, count=16):
    if destination.exists():
        raise FileExistsError("Preserve snapshot")
    original = fetch("data/ori_pqal.json")
    license_bytes = fetch("LICENSE")
    data = json.loads(original)
    if len(data) != 1000:
        raise ValueError("Pinned expert dataset population changed")
    public, gold = partition(data, count)
    private = destination / "private"
    publicdir = destination / "public"
    private.mkdir(parents=True)
    publicdir.mkdir()
    (private / "ori_pqal.json").write_bytes(original)
    (private / "gold.json").write_text(json.dumps(gold, indent=2) + "\n")
    (publicdir / "tasks.json").write_text(json.dumps(public, indent=2) + "\n")
    (destination / "LICENSE").write_bytes(license_bytes)
    manifest = {"complete": True, "commit": COMMIT, "source_url": BASE + "data/ori_pqal.json",
                "paper": "https://aclanthology.org/D19-1259/", "population": 1000, "count": count,
                "selection": "Lowest SHA256(salt+PMID), no label/topic/length/model-score filtering", "salt": SALT,
                "source_sha256": sha(original), "license_sha256": sha(license_bytes),
                "public_sha256": sha((publicdir / "tasks.json").read_bytes()),
                "gold_sha256": sha((private / "gold.json").read_bytes()),
                "snapshotter_sha256": sha(Path(__file__).read_bytes()),
                "limits": ["Oracle corresponding abstract, not a learned retrieval result",
                           "Abstract contexts exclude conclusions; LONG_ANSWER, human labels, prior predictions and metadata not public inputs",
                           "Public old dataset may be in pretraining; no contamination-free claim",
                           "Not official split/leaderboard score, clinical decision support, current medical consensus, legal or tax evidence"],
                "status": "Snapshot only. No model answers generated or competence established."}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    parser.add_argument("--count", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(snapshot(args.destination, args.count)))
