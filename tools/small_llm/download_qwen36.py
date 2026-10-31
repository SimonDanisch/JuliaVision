"""Pinned native checkpoint download; no Julia environment or package changes.

Only writes inside an explicitly owned download directory. Completed files are
SHA256-verified against the publisher's LFS metadata before being accepted.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.request

REPO = "Qwen/Qwen3.6-35B-A3B"
REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
RESERVE = 5 * 1024**3


def get_json(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.load(response)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    partial = path.with_suffix(path.suffix + ".partial")
    with open(partial, "x") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)


def prepare(directory):
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "download-source.json"
    identity = {"repo": REPO, "revision": REVISION}
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise RuntimeError("Download directory belongs to another source")
    elif any(directory.iterdir()):
        raise RuntimeError("Refusing unowned nonempty download directory")
    else:
        atomic_json(marker, identity)
    api = get_json(f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true")
    if api["sha"] != REVISION:
        raise RuntimeError("Publisher revision mismatch")
    small = {"config.json", "model.safetensors.index.json", "tokenizer.json",
             "tokenizer_config.json", "generation_config.json"}
    files = []
    for entry in api["siblings"]:
        name = entry["rfilename"]
        if name not in small and not name.endswith(".safetensors"):
            continue
        if Path(name).name != name:
            raise RuntimeError("Unexpected nested checkpoint path")
        lfs = entry.get("lfs", {})
        files.append({"name": name, "size": entry["size"], "sha256": lfs.get("sha256")})
    if not any(f["name"].endswith(".safetensors") for f in files):
        raise RuntimeError("No checkpoint files")
    manifest = {**identity, "files": sorted(files, key=lambda f: f["name"])}
    manifest_path = directory / "publisher-manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError("Publisher manifest drift")
    else:
        atomic_json(manifest_path, manifest)
    return manifest


def download(directory, entry):
    name, size, expected = entry["name"], entry["size"], entry["sha256"]
    target = directory / name
    if target.exists():
        if target.stat().st_size != size or (expected and sha256(target) != expected):
            raise RuntimeError(f"Existing checkpoint failed verification: {name}")
        print(f"Verified existing {name}", flush=True)
        return
    if shutil.disk_usage(directory).free < size + RESERVE:
        raise RuntimeError("Insufficient storage; preserving 5 GiB free")
    partial = directory / (name + ".partial")
    # A partial belongs to this exact pinned source. Restart only that file;
    # never truncate an accepted checkpoint or touch unrelated data.
    with urllib.request.urlopen(f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}", timeout=120) as response:
        h = hashlib.sha256()
        with open(partial, "wb") as stream:
            for block in iter(lambda: response.read(8 * 1024**2), b""):
                if shutil.disk_usage(directory).free < len(block) + RESERVE:
                    raise RuntimeError("Storage reserve reached during download")
                stream.write(block)
                h.update(block)
            stream.flush()
            os.fsync(stream.fileno())
    if partial.stat().st_size != size or (expected and h.hexdigest() != expected):
        raise RuntimeError(f"Downloaded checkpoint failed verification: {name}")
    os.replace(partial, target)
    print(f"Verified {name}: {size} bytes sha256={h.hexdigest()}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    manifest = prepare(args.directory)
    total = sum(entry["size"] for entry in manifest["files"])
    print(f"Pinned {REPO}@{REVISION}: {len(manifest['files'])} files, {total} bytes", flush=True)
    if args.manifest_only:
        return
    missing = sum(f["size"] for f in manifest["files"] if not (args.directory / f["name"]).exists())
    if shutil.disk_usage(args.directory).free < missing + RESERVE:
        raise RuntimeError("Insufficient space for checkpoint plus 5 GiB reserve")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda e: download(args.directory, e), manifest["files"]))


if __name__ == "__main__":
    main()
