"""Next fixed-hash16 questions + label-free1000-abstract retrieval corpus.
Reuses pinned original bytes; no second download, topic/label/length filtering,
training or inclusion of article titles/conclusions in indexed documents.
"""
import argparse
import json
from pathlib import Path
from snapshot_pubmedqa_pilot import partition, sha


def public_corpus(data):
    # Validate every record, not merely selected items.
    partition(data, count=len(data))
    return [{"qualified": pmid, "text": "\n\n".join(data[pmid]["CONTEXTS"])} for pmid in sorted(data)]


def next_cohort(data, previous_ids):
    first32, labels = partition(data, count=32)
    if {row["pmid"] for row in first32[:16]} != set(previous_ids):
        raise ValueError("Previous cohort drift")
    tasks = first32[16:32]
    return tasks, {row["pmid"]: labels[row["pmid"]] for row in tasks}


def snapshot(previous, destination):
    if destination.exists():
        raise FileExistsError("Preserve new snapshot")
    manifest_bytes = (previous / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    original = (previous / "private/ori_pqal.json").read_bytes()
    prior_public = (previous / "public/tasks.json").read_bytes()
    if sha(original) != manifest["source_sha256"] or sha(prior_public) != manifest["public_sha256"]:
        raise ValueError("Pinned source/first cohort drift")
    data = json.loads(original)
    if len(data) != 1000:
        raise ValueError("Complete public corpus required")
    tasks, gold = next_cohort(data, [r["pmid"] for r in json.loads(prior_public)])
    corpus = public_corpus(data)
    public = destination / "public"
    private = destination / "private"
    public.mkdir(parents=True)
    private.mkdir()
    for path, payload in ((public / "tasks.json", tasks), (public / "corpus.json", corpus), (private / "gold.json", gold)):
        path.write_text(json.dumps(payload, indent=2) + "\n")
    license_bytes = (previous / "LICENSE").read_bytes()
    if sha(license_bytes) != manifest["license_sha256"]:
        raise ValueError("License drift")
    (destination / "LICENSE").write_bytes(license_bytes)
    packet = {"complete": True, "count": 16, "corpus_documents": 1000, "commit": manifest["commit"],
              "source_url": manifest["source_url"], "source_sha256": sha(original), "license_sha256": sha(license_bytes),
              "previous_manifest_sha256": sha(manifest_bytes), "previous_public_sha256": sha(prior_public),
              "selection": "Same SHA256 salt/ranking, positions16:32, excludes entire prior16. No labels/topics/lengths/model scores select tasks.",
              "public_sha256": sha((public / "tasks.json").read_bytes()), "corpus_sha256": sha((public / "corpus.json").read_bytes()),
              "gold_sha256": sha((private / "gold.json").read_bytes()), "snapshotter_sha256": sha(Path(__file__).read_bytes()),
              "parent_snapshotter_sha256": sha(Path(__file__).with_name("snapshot_pubmedqa_pilot.py").read_bytes()),
              "scope": "New public-paper cohort after oracle pilot. Body-only source retrieval, no LONG_ANSWER, final_decision, prior predictions, titles or metadata in corpus. Not official leaderboard/contamination-free/current medical advice."}
    (destination / "manifest.json").write_text(json.dumps(packet, indent=2) + "\n")
    return packet


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("previous", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(snapshot(args.previous, args.destination)))
