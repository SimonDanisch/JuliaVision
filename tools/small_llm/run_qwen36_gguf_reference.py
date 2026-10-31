"""Same-GGUF Vulkan baseline under the Bosgame lock; no CPU-model fallback."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import os
from download_qwen36_gguf import NAME, SIZE, SHA256


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def main(directory, output, binary, device, trace=False):
    checkpoint = directory / NAME
    if checkpoint.stat().st_size != SIZE or sha(checkpoint) != SHA256:
        raise RuntimeError("Unverified quantized checkpoint")
    if output.exists():
        raise FileExistsError("Preserve first reference attempt")
    environment = dict(os.environ, VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/radeon_icd.json",
                       VK_DRIVER_FILES="/usr/share/vulkan/icd.d/radeon_icd.json")
    arguments=[str(binary.resolve()),str(checkpoint.resolve()),str(output.resolve()),device]
    if trace:
        arguments.append("trace")
    log=output.with_suffix(".log")
    log.parent.mkdir(parents=True,exist_ok=True)
    with open(log,"x") as stream:
        result=subprocess.run(arguments,check=True,env=environment,timeout=240,stdout=stream,stderr=stream)
    reference = json.loads((output / ("trace.json" if trace else "reference.json")).read_text())
    if not reference["complete"]:
        raise RuntimeError("Incomplete independent reference")
    provenance = dict(checkpoint_sha256=SHA256, executable_sha256=sha(binary),
                      backend="Vulkan Radeon; all layers/KV offloaded", source_sha256=sha(Path(__file__).with_name("reference_qwen36_gguf.cpp")),
                      logits_sha256=sha(output / "logits.f32"), returncode=result.returncode,
                      native_library_sha256={name: sha(binary.parent / "llama-vulkan-build" / "bin" / name)
                          for name in ("libllama.so", "libggml.so", "libggml-base.so", "libggml-vulkan.so")})
    with open(output / "provenance.json", "x") as stream:
        json.dump(provenance, stream, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--binary", type=Path, default=Path("tmp/qwen36-gguf-reference"))
    parser.add_argument("--device", default="Radeon 8060S")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    main(args.checkpoint_directory, args.output, args.binary, args.device, args.trace)
