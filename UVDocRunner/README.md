# UVDocRunner

Straightens curved document images using UVDoc on a Vulkan GPU through
DNNKernels/Mantle. The network predicts a backward deformation grid; Julia
resamples the original image at its full resolution. Input and output use
`img[x, y]`, with RGB colors. Output is `RGB{Float32}` in [0,1].

```julia
using UVDocRunner
model = uvdoc()                    # construct on the main GPU thread
flat = unwarp(model, page)         # same resolution as page
grid = deformation_grid(model, page)  # 31×45×2 normalized source coordinates
close(model)                      # after all pages have finished
```

The worker serializes inference and keeps GPU submission on its owning thread,
so calls from `Threads.@spawn` are supported. The model uses a 488×712 RGB
input. Preprocessing, grid interpolation and image sampling match Transformers'
`UVDocImageProcessor`: bilinear, `align_corners=true`, zero padding for image
coordinates outside the source. No Python is needed at runtime.

In AScanner:

```julia
using AScanner
dewarper = uvdoc()
scanner = start(cam; dewarper)
pdfs = stop!(scanner)
# Or reprocess an existing scan:
pdfs = make_pdfs("scans/my-scan"; dewarper)
close(dewarper)
```

Unwarping is opt-in: `dewarper=nothing` is the default. The scanner saves the
perspective-corrected capture, then applies UVDoc before cleaning and OCR.
Thus the PDF image and its text coordinates describe the same corrected page.
UVDoc estimates geometry; it does not guarantee that an already flat page
remains unchanged. Camera captures at the final mount still need evaluation.

## Export and validation

Code and weights: Apache-2.0. Model:
[PaddlePaddle/UVDoc_safetensors](https://huggingface.co/PaddlePaddle/UVDoc_safetensors),
the Transformers port of [UVDoc](https://github.com/tanguymagne/UVDoc).
7.9M parameters, fp32. The export reverses the first convolution's channels
so the Julia API accepts RGB; it checks against the unmodified BGR model.

```sh
uv run tools/export_uvdoc.py --reference-image curled-page.jpg
```

Then run `include("tools/make_artifacts.jl"); makeartifacts(["uvdoc"])` in
Julia. This binds a local artifact and prints the separate upload command.
`gen/graphs/uvdoc/provenance.json` records the resolved upstream revision.
The exporter writes PyTorch inputs, predicted grid and full-resolution output
to `reference.safetensors`. Tests compare preprocessing, GPU inference and
resampling separately, plus identity/reflection/border grids and concurrent
background calls. Run `include("UVDocRunner/test/runtests.jl")` in the monorepo
environment with a working GPU and local export references.

Validated on an AMD Radeon RX 7900 XTX with 15 runner tests, including two concurrent
requests. On the official example book photo (1100×708), the grid's maximum
absolute difference from PyTorch was 8.4e-7 and the output image's was 1.4e-4
on the [0,1] scale. Five warmed end-to-end calls took 18.8–19.8 ms, including
host preprocessing and full-resolution resampling. These timings exclude model
loading and compilation; other hardware and resolutions will differ.
