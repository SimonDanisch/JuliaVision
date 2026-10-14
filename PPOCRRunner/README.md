# PPOCRRunner

PP-OCRv6 medium on the GPU: the lines of text on a page, what they say, and
which way up the page is. Three networks from PaddleOCR, through the PyTorch
ports in Hugging Face Transformers:

| network | what it does | parameters | graph |
|---|---|---:|---|
| PP-OCRv6 medium det | marks every pixel inside a line of text (PP-LCNetV4, FPN neck, DB head) | 22M | static, 1280x1792 |
| PP-OCRv6 medium rec | reads one line, 48 pixels high (PP-LCNetV4, two SVTR blocks, CTC over 18710 characters) | 19M | symbolic in the line's width |
| PP-LCNet x1.0 doc_ori | 0, 90, 180 or 270 degrees | 1.7M | static, 224x224 |

License: **Apache-2.0**, code and weights. Latin script including ä ö ü ß €,
and Chinese, Japanese, Korean and more.

```julia
using PPOCRRunner
m = ppocr()
degrees = orientation(m, img)          # img[x, y], any AbstractRGB
for line in read_lines(m, img)         # on the upright page
    println(line.text, " ", line.box)  # and line.words: (text, x0, x1)
end
```

## Measured

On a Radeon RX 7900 XTX (RADV), 2026-10-10, against PyTorch on the same inputs:

| | result |
|---|---|
| detector map, 1280x1792 | within 7.9e-4 of PyTorch, 93 ms |
| recogniser, one line | same character at every time step, best probability within 3.6e-7, about 20 ms |
| orientation | logits within 4.8e-7, 10 ms; 8 of 8 rotations of two test pages right |
| an A4 letter at 300 dpi | 125 of 125 words, 0.57 s |
| a photographed menu card | 22 of 22 words incl. "HOCHZEITSMENÜ", where Tesseract 5 gets 21 |

Tesseract 5 on the same letter takes 2.0 s on one CPU core. On degraded copies
of the card (half resolution, blur, JPEG at quality 25, a shadow across it)
PP-OCRv6 stayed at 100% and Tesseract fell to 91%, and to 4.5% under the shadow.

## What the port needed in DNNKernels

Nothing in the op set; four fixes, each with a test in
`DNNKernels/test/test_ppocr_regressions.jl` or `test_maxdim.jl`:

  * **Grouped and depthwise convolutions** had no declared lowering. PP-LCNetV4
    has one between every pair of pointwise layers; they now run `conv2d`, one
    output per thread.
  * **Batch norm folded into a transposed convolution** scaled its input
    channels: the output-channel axis of `(kx, ky, Cout, Cin)` is the third.
    With `Cin == Cout` nothing failed and the detector's map was 0.86 off.
  * **`max.dim` without `keepdim`** read the first slice for every output: the
    recogniser returned the blank class at every step.
  * **`mean.dim` over a whole plane** gave each output one thread. The
    squeeze-and-excitation pools were 912 ms of a 1011 ms detector; one
    workgroup per output brings the detector to 93 ms.

## Not in the graphs

Scaling a page into the detector's input, turning its map into boxes (connected
components and DB's unclip), cutting lines out at 48 pixels, CTC decoding, and
the orientation crop all run on the host. The recogniser's width is rounded up
to one of eight planned widths (`REC_WIDTHS`), so a page brings no new kernels.

## Making the assets

```
uv run --project <env with torch + transformers> tools/export_ppocrv6.py --device cpu \
    --reference-image <a page>
julia --project=<env> tools/make_artifacts.jl ppocrv6_det ppocrv6_rec ppocr_docori
```

CPU export is fine: the recogniser's two attention blocks decompose into `bmm`
and `_softmax`, over a few hundred tokens. `--verify` writes small graphs with
every node's PyTorch result for `DNNKernels.verifygraph`. Give it real text:
a blank line verified clean while `max.dim` was still wrong.
