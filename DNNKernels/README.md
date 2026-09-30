# DNNKernels

The graph runtime under every model in this repository: it loads a PyTorch
model exported with `torch.export`, rewrites the ATen graph, and declares it into
a [Mantle](https://github.com/SimonDanisch/Lava.jl) graph, which places the
memory, aliases the transients and inserts the barriers before anything runs.
The same graph then replays from a recording on every call, and it runs on
Vulkan through Lava, on Metal, and on ROCm where Mantle has those backends.

## Running an exported model

```julia
using DNNKernels
using DNNKernels: toback
import Mantle

graph   = loadgraph("depthanything.json")          # what tools/export_*.py write
weights = readsafetensors("weights.safetensors")
backend = Mantle.defaultbackend()
m = Model(Dict("depthanything" => graph), weights; backend)

x = toback(backend, rand(Float32, 518, 518, 3, 1))  # torch's (1, 3, H, W), reversed
out, = DNNKernels.call(m, "depthanything", x; dims = (;))
```

Shapes are torch's reversed: `readsafetensors` and `loadgraph` both flip the
extents, so a `(N, C, H, W)` tensor is a `(W, H, C, N)` array and needs no
permute. `dims` binds the graph's symbolic extents, if it has any. The first call
with a given `dims` builds and records a plan; every later one replays it.
`Model` holds device buffers; drop it, or `DNNKernels.releaseplans!(m)`, when done.

Each `*Runner` package is this plus a precompile workload, so that both halves
of the cold start (Julia's inference and the SPIR-V compile) are paid at
precompile time: 48.2 s to 3.3 s for SAM 2's first click.

## What `Model` does to a graph

Host-side rewrites, each its own file and each counted in the `@info` it prints:
`foldbatchnorm`, `foldrelu`, `foldconvpad`, `halfconvs` (fp16 convolution
operands in an fp32 graph, opt-in), `hoistcasts`, `hoistpermutes`,
`hoistconstants`, `fuseqkv`, `foldoutcasts` and `foldincasts`, `dropclones`,
`fuseswiglu` and `fuseswiglumm`, `fusegroupedrms`, `foldcacheupdate` (a KV cache
updated in place), `fuserope`, `fusepairrope`, `fusermsrope`, `dropdead`. Then
the weights go to the device, with `quantize = true` storing large ones as int8,
and the fusions that need them there: `fuseattention` (bmm, softmax, bmm into
flash attention), `fuseops` (elementwise chains into one kernel),
`fusemaskedattention`, `foldepilogue` and `foldpremap`.

Every op a pass can create has an `emitop!`; `test_foldoutcasts.jl` fails if one
does not, which is the lesson `fused.maskedattention` taught (see below).

## Checking a port

`verifygraph(graph, refs, weights; dims, backend)` runs the graph node by node
against PyTorch's own activations and reports the first mismatch and how much it
amplifies downstream. The `*-refs` artifacts hold those activations for SAM 2 and
MatAnyone; `tools/verify_*.jl` do the same for the others. What the runners
claim, from those tools:

| model | against PyTorch |
|---|---|
| Depth Anything V2 Small | 4.2e-5 |
| RIFE 4.26 | 3.3e-4 max, 4.9e-7 mean |
| Neural 3D LUT | 3.6e-7 (table), 3.0e-7 (apply) |
| BasicVSR++ | 1.1e-5 max over five frames |
| Whisper encoder | relative rms 6.3e-5 |
| Kokoro | correlation > 0.95 on the audio |

## The kernels

Kernels are plain functions written against KernelInterface
(`KI.get_global_id()`), not `@kernel` blocks, and `Mantle.dispatch!` takes the
function itself and compiles it for the backend the graph is on. The fast paths
are cooperative-matrix GEMM, flash attention and int8 GEMM with the dequant in
the epilogue. The portability that is verified is across Vulkan devices: one
SPIR-V module on an RTX 4000 Ada (subgroup 32, `VK_NV_cooperative_matrix2`) and
a Radeon 8060S (RDNA 3.5, subgroup 64, KHR cooperative matrix only), with the
tile size and cooperative-matrix support asked per device. There is no
vendor-conditional code.

Compiled kernels are frozen to disk per `KERNELS_VERSION`, shared by every model:
**bump it after editing a kernel body**, because the cache is keyed on the
function and argument types, not the source.

Quantised weights: int8 (`quantize = true`), ConvRot int8 and W4A8 for the
Qwen-Image denoiser, and PTQ1 ternary with a GGUF reader for Bonsai.

## Recent fixes worth knowing

  * **`fused.maskedattention` (2026-09-30).** Its only implementation was the
    eager arm, deleted with the eager op library on 2026-09-23, while
    `fusemaskedattention` kept creating the op. Every fp16 graph with a masked
    attention was refused from then on; Horizon 32B could not run at all. It is
    declared now, with the eager arm's plan choice, and
    `test/test_masked_flash.jl` runs each plan family against the definition.
  * **`topk` indices (2026-09-30).** They were stored through the value dtype,
    so Hunyuan3D's int64 indices over fp16 went through `Int64(::Float16)` and an
    `InexactError` path inside the shader.
  * **`fused.rope` layouts (2026-09-29).** Tables broadcast against `x` the way
    the graph broadcasts them, which the Qwen3-VL vision tower's layout needed.

## Tests

`]test DNNKernels`: about 29,000 assertions, most of them ops against their
definitions written out, plus the declared-coverage sweep over every graph in
the local depot.
