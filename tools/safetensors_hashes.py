"""Print SHA-256 per tensor, over its raw bytes (no dtype conversion)."""
import hashlib
import json
import struct
import sys


def tensor_hashes(path):
    with open(path, "rb") as stream:
        length, = struct.unpack("<Q", stream.read(8))
        header = json.loads(stream.read(length))
        base = stream.tell()
        result = {}
        for name, spec in header.items():
            if name == "__metadata__":
                continue
            lo, hi = spec["data_offsets"]
            stream.seek(base + lo)
            remaining = hi - lo
            digest = hashlib.sha256()
            while remaining:
                block = stream.read(min(8 * 1024 * 1024, remaining))
                if not block:
                    raise ValueError(f"truncated tensor {name}")
                digest.update(block)
                remaining -= len(block)
            result[name] = digest.hexdigest()
        return result


if __name__ == "__main__":
    json.dump(tensor_hashes(sys.argv[1]), sys.stdout)
