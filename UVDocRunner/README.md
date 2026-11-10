# UVDocRunner

Straightens curved document images using UVDoc on a Vulkan GPU through
DNNKernels/Mantle. The network predicts a backward deformation grid; Julia
resamples the original image at its full resolution. Input and output use
`img[x, y]`, with RGB colors. Output is `RGB{Float32}` in [0,1].

```julia
using UVDocRunner
model = uvdoc()                    # on the thread that owns the device
flat = unwarp(model, page)         # RGB{Float32}, same resolution as page
close(model)
```

Resizing into the network's 488×712 input, the network and the warp along its
grid all run on the device; only the flattened page comes back. A caller with
its page on the device already writes it into
`UVDocRunner.pageinput!(model, size).pixels`, calls
`UVDocRunner.unwarp!(model, page)` and reads `page.output`. Preprocessing, grid
interpolation and image sampling match Transformers' `UVDocImageProcessor`:
bilinear, `align_corners=true`, zero padding for image coordinates outside the
source. No Python is needed at runtime. Use a model on the thread that made
its device: Mantle submits from there.

UVDoc estimates geometry; it does not guarantee that an already flat page
remains unchanged.

## Assets and validation

One artifact, `uvdoc`, downloaded on first use. Code and weights: Apache-2.0.
Model: [PaddlePaddle/UVDoc_safetensors](https://huggingface.co/PaddlePaddle/UVDoc_safetensors),
the Transformers port of [UVDoc](https://github.com/tanguymagne/UVDoc).
7.9M parameters, fp32. The first convolution's channels are reversed so the
Julia API takes RGB; the export is checked against the unmodified BGR model.

Tests compare the device's preprocessing, the network and the warp with
PyTorch's inputs, predicted grid and full-resolution output separately, plus
identity, reflection and border grids written where the network writes its own.
