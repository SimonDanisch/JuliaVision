"""Download only the pinned19.89GB Q4_K_S checkpoint; never native BF16."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.request

REPO = "mradermacher/Qwen3.6-35B-A3B-GGUF"
REVISION = "ac96549d2e8b902f1a772083a33d13b5ee01fe3c"
NAME = "Qwen3.6-35B-A3B.Q4_K_S.gguf"
SIZE = 19889904096
SHA256 = "97cc5853d3c163c9cb81782ffc8ee18d909c67f8a3c6b4816b98f1103125c6c6"
URL = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{NAME}"


def main(directory, header_only):
    directory.mkdir(parents=True, exist_ok=True)
    source = dict(repo=REPO, revision=REVISION, name=NAME, size=SIZE, sha256=SHA256)
    marker = directory / "source.json"
    if marker.exists():
        if json.loads(marker.read_text()) != source:
            raise RuntimeError("Wrong source directory; refusing overwrite")
    elif any(directory.iterdir()):
        raise RuntimeError("Unowned nonempty destination")
    else:
        with open(marker, "x") as stream:
            json.dump(source, stream, indent=2)
    name = NAME + ".header" if header_only else NAME
    target = directory / name
    if target.exists():
        raise FileExistsError(f"Preserve existing download: {target}")
    size = 32 * 1024**2 if header_only else SIZE
    if shutil.disk_usage(directory).free < size + 5 * 1024**3:
        raise RuntimeError("Need download size plus5GiB disk reserve")
    request = urllib.request.Request(URL)
    if header_only:
        request.add_header("Range", f"bytes=0-{size-1}")
    h = hashlib.sha256()
    count = 0
    partial = target.with_suffix(target.suffix + ".partial")
    with urllib.request.urlopen(request, timeout=120) as response, open(partial, "xb") as stream:
        if header_only and response.status != 206:
            raise RuntimeError("Publisher did not honor bounded header range")
        while count < size:
            block = response.read(min(8 * 1024**2, size-count))
            if not block:
                raise RuntimeError("Truncated download")
            if shutil.disk_usage(directory).free < len(block) + 5 * 1024**3:
                raise RuntimeError("Disk reserve reached")
            stream.write(block); h.update(block); count += len(block)
            if count % (1024**3) < len(block):
                print(f"Downloaded {count}/{size} bytes", flush=True)
        stream.flush(); os.fsync(stream.fileno())
    if not header_only and h.hexdigest() != SHA256:
        raise RuntimeError("Publisher SHA256 mismatch; not accepting file")
    os.replace(partial, target)
    print(json.dumps(dict(complete=True, header_only=header_only, bytes=count, sha256=h.hexdigest(), path=str(target))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--header-only", action="store_true")
    args = parser.parse_args()
    main(args.directory, args.header_only)
