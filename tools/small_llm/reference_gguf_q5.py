"""Independent Q5_K decoder fixtures using llama.cpp's Python GGUF package."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def main(output):
    if output.exists():
        raise FileExistsError("Preserve first packed-format oracle")
    sys.path.insert(0, "/home/sim/Programmieren/llama.cpp/gguf-py")
    from gguf.quants import dequantize
    from gguf.constants import GGMLQuantizationType
    from safetensors.numpy import save_file
    rng = np.random.default_rng(365)
    values = {}
    for k in (256, 512, 1024):
        rows, experts = 17, 4
        packed = rng.integers(0, 256, (rows * experts * (k // 256), 176), dtype=np.uint8)
        packed[:, :2] = np.array([0.025], dtype=np.float16).view(np.uint8)
        packed[:, 2:4] = np.array([0.01], dtype=np.float16).view(np.uint8)
        packed = packed.reshape(rows * experts, (k // 256) * 176)
        expected = dequantize(packed, GGMLQuantizationType.Q5_K)
        values[f"q5.{k}.bytes"] = packed.copy()
        values[f"q5.{k}.float"] = expected.astype(np.float32)
        values[f"q5.{k}.input"] = rng.normal(size=(3, 2, k)).astype(np.float32)
    output.mkdir(parents=True)
    save_file(values, str(output / "reference.safetensors"))
    report = dict(generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  decoder="llama.cpp gguf.quants.dequantize Q5_K",
                  decoder_sha256=hashlib.sha256(Path(sys.modules[dequantize.__module__].__file__).read_bytes()).hexdigest(),
                  tensor_sha256=hashlib.sha256((output / "reference.safetensors").read_bytes()).hexdigest())
    (output / "reference.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(complete=True, cases=3)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
