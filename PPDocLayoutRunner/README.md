# PPDocLayoutRunner

PP-DocLayoutV3 on the GPU: a page's regions, what each is, and the order to
read them in. The 33M-parameter detector from PaddleOCR (an RT-DETR with a
reading-order head) sees the page at 800x800 and reports regions of 25 kinds:
text, document and paragraph titles, tables, images, charts, formulas, headers,
footers, footnotes, seals, page numbers and more.

```julia
using PPDocLayoutRunner
m = doclayout()
for r in regions(m, img)          # img[x, y], any colour; in reading order
    println(r.kind, " ", r.score, " ", r.box)   # box in page pixels
end
close(m)
```

The resize into the network runs on the device (`GPUFiltering`, bicubic and
saturated, as the `cv2.resize` the model was trained with); only 300 scores,
boxes and the 300x300 order logits come back. Choosing the regions above the
threshold and ranking them by the order head's votes is a few hundred values on
the host, as upstream's post-processing. A caller with its page on the device
already writes it into `PPDocLayoutRunner.pageinput!(m, size).pixels`.

## Measured

On a Radeon RX 7900 XTX (RADV), 2026-10-15, against PyTorch on the same input
tensor: the same regions kept, class logits within 0.02 and boxes within half
a pixel at 800x800, the same reading order. On a rendered A4 invoice at 300 dpi
the runner finds the same 15 regions as PyTorch's pipeline, kinds and order
included, in 0.11 s including the upload.

The model's own `topk` (300 of 13125 anchors) needed a selection for long
axes in DNNKernels (`topk_large_kernel!`: radix select, compaction, ranking in
one workgroup per slice), and `pow.Scalar`, `min.dim` and `any.dims`.

## Assets

Code and weights: Apache-2.0, `PaddlePaddle/PP-DocLayoutV3_safetensors`
through Hugging Face Transformers.

```sh
uv run tools/export_ppdoclayout.py --reference-image page.png
julia --project=<env> tools/make_artifacts.jl ppdoclayoutv3
```
