"""
PyTorch's own time for one Whisper large-v3-turbo ENCODER forward, on this card.

    uv run tools/baseline_whisper.py            # fp16, what we ship
    uv run tools/baseline_whisper.py --fp32

The other half of `tools/bench_all.jl whisper-encode`. A "≥ PyTorch" target needs
a denominator measured the same way on the same card — GUARDRAILS §5, do not rank
against an unmeasured denominator.

Matched to the Julia side as closely as the two runtimes allow, the same way
`baseline_depthanything.py` and `sam2_pytorch_baseline.py` are:

  * the encoder ALONE, not `generate` — that is what `WhisperRunner.encode` runs,
    and timing the autoregressive loop here would compare different work;
  * the same 30 s window shape, `(1, 128, 3000)`, from the reference dump when it
    is on disk so both sides see identical numbers, else zeros;
  * **the same dtype**. fp16 is the default because that is what the artifact now
    ships; `--fp32` for the other row. TF32 is left OFF in fp32 so PyTorch is not
    handed the tensor cores for every matmul while Lava runs true fp32, which
    would measure a dtype choice and call it an engine gap;
  * `torch.cuda.synchronize` around a batch rather than each iteration, so the
    sync is not what is being timed;
  * a warm-up before anything is recorded, and the median reported, not the min.

**No `torch.compile`.** The graph Lava runs came out of `torch.export`, which is
eager PyTorch's own decomposition. Comparing against a fused Inductor build would
be comparing a compiler we do not have against a runtime we do.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import find_root  # noqa: E402

ROOT = find_root()


def load_mel(dev, dtype):
    """The reference window if it is on disk, else zeros of the right shape."""
    for d in ("whisper-fp16", "whisper"):
        p = ROOT / "gen" / "graphs" / d / "refs.safetensors"
        if p.is_file():
            from safetensors.torch import load_file
            r = load_file(str(p))
            for k in ("whisper/in0",):
                if k in r:
                    # The dump is in Lava's reversed layout, (3000, 128, 1);
                    # torch wants (1, 128, 3000).
                    t = r[k]
                    if t.shape == (3000, 128, 1):
                        t = t.squeeze(-1).T.unsqueeze(0)
                    return t.to(dev, dtype)
    return torch.zeros(1, 128, 3000, device=dev, dtype=dtype)


def main(fp32, n, reps):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = "cuda"
    dtype = torch.float32 if fp32 else torch.float16

    from transformers import WhisperForConditionalGeneration
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3-turbo", torch_dtype=dtype).to(dev).eval()
    enc = model.model.encoder

    mel = load_mel(dev, dtype)
    with torch.inference_mode():
        for _ in range(3):                       # warm the clock and the caches
            enc(mel)
        torch.cuda.synchronize()

        times = []
        for _ in range(reps):
            torch.cuda.synchronize()
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            for _ in range(n):
                enc(mel)
            t1.record()
            torch.cuda.synchronize()
            times.append(t0.elapsed_time(t1) / n)

    out = {"device": torch.cuda.get_device_name(0),
           "dtype": str(dtype), "shape": list(mel.shape),
           "encode_ms": {"min": min(times), "p50": statistics.median(times)}}
    print(json.dumps(out, indent=1))
    print(f"\nwhisper encoder {'fp32' if fp32 else 'fp16'}: "
          f"{statistics.median(times):.1f} ms per 30 s window (p50, TF32 off)")


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--fp32", action="store_true")
    a.add_argument("-n", type=int, default=5)
    a.add_argument("--reps", type=int, default=7)
    a = a.parse_args()
    main(a.fp32, a.n, a.reps)
