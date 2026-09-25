"""Qwen-Image 2.1 VAE decoder: PyTorch's answer to the latents Julia just made.

    julia --project=<repo-root> tools/parity_qwenimage_vae.jl
    benchmark/torch.sh tools/parity_qwenimage_vae.py

The second half of `parity_qwenimage_vae.jl`; that file's header says why the
latents come from twenty real denoise steps rather than from the exporter's own
`vae_reference.safetensors`, and carries the measured numbers.

Writes `tmp/parity_torch.bin` so the Julia side can render the two against each
other.
"""

import os, sys
import numpy as np, torch

# After `import torch`, exactly as benchmark/torch_models.py does it: the ROCm
# interpreter has torch and little else, and the project .venv that supplies
# safetensors/diffusers also contains a CUDA torch that must not shadow it.
for _p in filter(None, os.environ.get("JULIAVISION_TORCH_EXTRA_SITE", "").split(os.pathsep)):
    if _p not in sys.path:
        sys.path.append(_p)

sys.path.insert(0, "tools")
from artifacts import artifact
from export_qwenimage21 import import_qwen21, NormalizedVAEDecoder

lw = lh = 16
# Julia wrote (w,h,1,c,b) column-major, which is byte-identical to C-order
# (b,c,1,h,w) — the torch layout — so this needs no transpose.
lat = np.fromfile("tmp/parity_latents.bin", dtype=np.float32).reshape(1, 64, 1, lh, lw)
ours = np.fromfile("tmp/parity_ours.bin", dtype=np.float32).reshape(1, 4, 1, 256, 256)[:, :, 0]

_transformer_module, module = import_qwen21(artifact("diffusers-src") / "src")
vae = module.AutoencoderKLQwenImage21.from_pretrained(
    "Qwen/Qwen-Image-2.1", subfolder="vae", torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True).eval()
with torch.no_grad():
    ref = NormalizedVAEDecoder(vae).eval()(torch.from_numpy(lat).to(torch.bfloat16)).float().numpy()

d = np.abs(ours - ref)
sc = float(np.abs(ref).max())
print(f"shapes  ours {ours.shape}  torch {ref.shape}")
print(f"NaN     ours {int(np.isnan(ours).sum())}   torch {int(np.isnan(ref).sum())}")
print(f"max|d|  {d.max():.5f}   ({100*d.max()/sc:.3f}% of the reference's {sc:.4f} range)")
print(f"mean|d| {d.mean():.6f}      RMS {np.sqrt((d**2).mean()):.6f}")
x, y = ours.ravel() - ours.mean(), ref.ravel() - ref.mean()
print(f"corr    {float((x*y).sum()/np.sqrt((x*x).sum()*(y*y).sum())):.7f}")
for c in range(4):
    dc = d[0, c]
    print(f"  ch{c}   max {dc.max():.5f}  mean {dc.mean():.6f}")
np.fromfile  # keep ruff quiet
ref.astype(np.float32).tofile("tmp/parity_torch.bin")
