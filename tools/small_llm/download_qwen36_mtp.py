"""Download only Qwen3.6-35B-A3B's multi-token prediction layer (`mtp.*`, 1.6 GB of
BF16) from the publisher's native checkpoint, by byte ranges of the two shards
that hold it, into one safetensors file per shard. Each file is checked against
the SHA-256 of the file cut from the whole shards, themselves checked against
the publisher's shard hashes when this was pinned.

    python -I download_qwen36_mtp.py <workspace>/models/Qwen3.6-35B-A3B-mtp
    python -I download_qwen36_mtp.py <dir> --plan-only   # headers only, no tensors
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import urllib.request

REPO = "Qwen/Qwen3.6-35B-A3B"
REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
PREFIX = "mtp."
# Shard => (output name, size, SHA-256 of the output).
SHARDS = {
    "model-00025-of-00026.safetensors": ("mtp-from-00025.safetensors", 1073741944,
                                         "8e73e5ac8eb0e473533582433bfd9f4183accca840690235b7abae9a4ba273d1"),
    "model-00026-of-00026.safetensors": ("mtp-from-00026.safetensors", 615541680,
                                         "856b6ed35877d5aaf40456f20de89a4a2f2f53e8b66e2b59d99a016d75f0b977"),
}


def url(name):
    return f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"


def fetch(name, first, last):
    """Bytes first..last (inclusive) of a checkpoint file."""
    request = urllib.request.Request(url(name))
    request.add_header("Range", f"bytes={first}-{last}")
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status != 206:
            raise RuntimeError(f"{name}: the publisher did not honor the byte range")
        data = response.read()
    if len(data) != last - first + 1:
        raise RuntimeError(f"{name}: {len(data)} bytes for a {last - first + 1}-byte range")
    return data


def plan(shard):
    """The kept tensors' header for the output file, and each tensor's absolute
    byte range in the shard, in shard order: the layout the whole-shard cut wrote."""
    (hlen,) = struct.unpack("<Q", fetch(shard, 0, 7))
    header = json.loads(fetch(shard, 8, 8 + hlen - 1))
    keep = sorted(((k, v) for k, v in header.items() if k != "__metadata__" and k.startswith(PREFIX)),
                  key=lambda kv: kv[1]["data_offsets"][0])
    if not keep:
        raise RuntimeError(f"{shard} holds no {PREFIX} tensor")
    newheader, at, ranges = {}, 0, []
    for k, v in keep:
        a, b = v["data_offsets"]
        newheader[k] = {"dtype": v["dtype"], "shape": v["shape"], "data_offsets": [at, at + (b - a)]}
        ranges.append((8 + hlen + a, 8 + hlen + b))
        at += b - a
    blob = json.dumps(newheader, separators=(",", ":")).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)
    return struct.pack("<Q", len(blob)) + blob, ranges


def main(directory, plan_only):
    directory.mkdir(parents=True, exist_ok=True)
    source = dict(repo=REPO, revision=REVISION, prefix=PREFIX,
                  files={out: dict(shard=shard, size=size, sha256=sha) for shard, (out, size, sha) in SHARDS.items()})
    marker = directory / "source.json"
    if marker.exists():
        if json.loads(marker.read_text()) != source:
            raise RuntimeError("Wrong source directory; refusing overwrite")
    elif any(directory.iterdir()):
        raise RuntimeError("Unowned nonempty destination")
    else:
        with open(marker, "x") as stream:
            json.dump(source, stream, indent=2)
    for shard, (out, size, sha) in SHARDS.items():
        target = directory / out
        if target.exists():
            print(f"{target} exists; keeping it", flush=True)
            continue
        head, ranges = plan(shard)
        if len(head) + sum(b - a for a, b in ranges) != size:
            raise RuntimeError(f"{shard}: the planned file is not the pinned {size} bytes")
        if plan_only:
            print(json.dumps(dict(shard=shard, header_bytes=len(head), tensors=len(ranges), size=size)), flush=True)
            continue
        if shutil.disk_usage(directory).free < size + 5 * 1024**3:
            raise RuntimeError("Need download size plus 5 GiB disk reserve")
        h = hashlib.sha256(head)
        partial = target.with_suffix(target.suffix + ".partial")
        with open(partial, "xb") as stream:
            stream.write(head)
            for a, b in ranges:
                at = a
                while at < b:
                    block = fetch(shard, at, min(b, at + 256 * 1024**2) - 1)
                    stream.write(block); h.update(block); at += len(block)
                print(f"{out}: {stream.tell()}/{size} bytes", flush=True)
            stream.flush(); os.fsync(stream.fileno())
        if h.hexdigest() != sha:
            os.remove(partial)
            raise RuntimeError(f"{out}: SHA-256 mismatch; not accepting file")
        os.replace(partial, target)
        print(json.dumps(dict(complete=True, path=str(target), bytes=size, sha256=sha)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    main(args.directory, args.plan_only)
