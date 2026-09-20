# QwenImageRunner

Qwen-Image 2.1 text-to-image and image-editing support for JuliaVision.

The host-side 2.1 pipeline contract is implemented: official architecture
constants, unpatched stride-16 latent flattening, resolution-dependent FlowMatch scheduling, Euler
updates, and separate transformer/text-encoder/VAE graph discovery.

The end-to-end GPU port is in progress. The compact Comfy-Org files require two
new weight paths before they are usable:

- The 7.26 GB denoiser is tensor-wise INT8 with ConvRot, not ordinary symmetric
  int8.
- The 6.31 GB text encoder is asymmetric W4A8 with a 16-value codebook, FP8
  per-group scales, per-channel scales, and ConvRot.

Until those formats are implemented in DNNKernels, the exporter uses the
official BF16 model as the numerical reference. Exported assets can be selected
with `JULIA_QWENIMAGE21_ASSETS`.

The denoiser exporter has a download-free smoke mode. Current Diffusers main is
needed until its Qwen-Image 2.1 implementation reaches a release. Its current
dependency floor is `huggingface-hub>=1.31` and `transformers>=5.17`:

```sh
.venv/bin/python tools/export_qwenimage21.py --smoke \
    --diffusers-source /path/to/diffusers/src
```

Upstream model: <https://huggingface.co/Qwen/Qwen-Image-2.1>  
Compact checkpoint: <https://huggingface.co/Comfy-Org/Qwen-Image-2.1>  
License: Qwen Research License Agreement.
