# OIDNRunner

Native JuliaVision/Mantle inference for Intel Open Image Denoise **v2.5.1**.
This is an independent Apache-2.0 port using the official, unmodified weights.
There is no OIDN C/C++ library in the inference path; the released CPU library
is used only to generate test references.

Includes the high-quality large HDR network and **both auxiliary prefilters**:
albedo and normals. HDR auto-scaling, perceptual transfer functions, 16-pixel
alignment, padding and overlapping tiles follow upstream. On matrix-capable GPUs
the default uses Float16 tensors and weights with Float32 matrix accumulation,
matching upstream HIP's precision policy. Float32 remains available explicitly;
other devices default to Float32. Inputs and outputs are Float32 linear HDR;
the perceptual transfer protects HDR range outside the network, so values above
one are retained. No weight quantization is used.
This is spatial, not temporal, denoising.

## Inference path

Float16 (the default on matrix-capable GPUs) runs the port of upstream's HIP
path in `src/unet.jl` and `src/gpupipeline.jl`; explicit Float32 keeps the
DNNKernels graph as the exact-reference mode.

- **Layout as upstream HIP:** HWC activations with channels padded to 8, OHWI
  weights. Each convolution is an implicit GEMM with both operands contiguous
  along `k`, so a WMMA fragment loads in two 16-byte reads. Activations and
  weights are staged in 16-byte vectors: 8 x fp16 on ROCm, a `uvec4` on Vulkan
  (SPIR-V vectors stop at four components), which Lava's `loadwu4` lets a
  cooperative-matrix load read back as fp16 out of shared memory.
- **Branch-free gather:** one load from a clamped address, then a select for the
  zero padding. Written with `&&`/`?:` it became control flow in SPIR-V, and
  RADV had to wait for each of the eight prefetch loads before the other side
  of the branch zeroed the same registers.
- **Fused like CK and then some:** bias is the accumulator's initial value;
  ReLU and the fp16 store happen in registers. The decoder's nearest upsample
  and channel concatenation are folded into the convolution's gather, so neither
  is materialised.
- **Tiling:** generated kernel family (`CONV_TILINGS`), chosen per output-channel
  count (`CONV_TABLE`) from a sweep over every layer at 1080p
  (`BlackholeImage/tools/oidn_conv_sweep.jl`). Upstream's CK instance gives each
  wave 16 channels x 128 pixels; a 32 x 64 wave block with one workgroup owning
  all output channels is 20-45% faster here (31-35 TFLOPS from 64 channels up).
  The 3-channel head is padded to 16 outputs, not 32.
- **On the device end to end:** radiance and guides are uploaded once; input
  transfer functions and tile placement, the albedo and normal prefilters, the
  HDR network and the output transfer all run on the GPU; one download. Only
  the 16x16-bin autoexposure stays on the host.

The ROCm backend is preferred when AMDGPU is loaded (`preferredbackend()`), the
platform default (Vulkan) otherwise. The same kernels serve both.

## Build the assets

From the workspace root, with the reference clone at \`dev/oidn\`:

    uv run --no-project --python 3.12 --index https://download.pytorch.org/whl/cpu \
      --with torch==2.9.1+cpu --with safetensors --with numpy \
      python dev/JuliaVision/tools/export_oidn.py --checkout dev/oidn \
      --output dev/JuliaVision/gen/graphs/oidn
    julia --project=. dev/JuliaVision/tools/make_oidn_artifacts.jl

The exporter verifies the Git LFS weight SHA256 values. Provenance pins code
commit \`6602ee2ca38a1a2a02135beed8f6e68eed630180\` and weight commit
\`28883d1769d5930e13cf7f1676dd852bd81ed9e7\`.
The packer installs an immutable local artifact and creates
\`oidn-models.tar.gz\`. The current artifact is hosted in the repository's
existing \`assets-v1\` release; Julia downloads it on first use and verifies
the archive SHA256 and artifact tree hash. Workers need no manual model copy.
Repacking unchanged content preserves its published download declaration.
For new content, upload the archive first, then supply its actual immutable
URL as the packer's second argument to bind that archive's SHA256. The packer
does not create releases or upload assets itself.

## Use

    using OIDNRunner, Mantle
    d = OIDNDenoiser() # installed artifact, or pass an explicit export directory
    # Exact-reference mode: OIDNDenoiser(; precision=Float32, tilesize=768).
    # Default 2048-pixel tiles avoid redundant overlap at 1080p. Lower tilesize
    # trades extra processing for lower GPU memory use.
    filtered = denoise(d, color; albedo, normal)
    # Matrices: RGB Float32 radiance/albedo, Vec3 Float32 world-space normals.
    # Also accepts planar (width,height,3) arrays; output has that planar shape.
    clean = prefilter(d, albedo, normal)
    filtered = denoise(d, color; albedo=clean.albedo, normal=clean.normal, clean_aux=true)
    OIDNRunner.release!(d)

Loading VideoEditor activates the optional extension. \`OIDNDenoiseEffect\`
then lazily uses the installed model artifact. \`install!(d)\` selects a
specific denoiser instance. Strength and guide-prefiltering are editable,
non-destructive effects; the stored source radiance/guides never get filtered
in place. Invalid legacy guides are refused rather than silently used.
VideoEditor's \`refreshguides!(clip, sourceframe)\` upgrades guides with primary
rays only, retaining the original radiance and coverage.

## Verification

Generate references with \`tools/reference_oidn.py\` and the released upstream
v2.5.1 library (the library is a test-only dependency):

    ... reference_oidn.py --library /path/libOpenImageDenoise.so \
      --models dev/JuliaVision/gen/graphs/oidn/inputs \
      --output dev/JuliaVision/gen/graphs/oidn/pipeline-reference.safetensors
    ... reference_oidn.py ... --width 815 --height 47 \
      --output dev/JuliaVision/gen/graphs/oidn/tiled-reference.safetensors

Run \`OIDNRunner/test/runtests.jl\` in the workspace environment.
In explicit Float32 mode on Bosgame's Radeon 8060S, all three networks match PyTorch within **1.4e-6**
maximum absolute error. The complete prefiltered HDR pipeline matches released
OIDN with **1.5e-6 relative RMS**, or **1.9e-6** on the overlapping-tile test.
These are reproducible fixtures, not a guarantee on arbitrary images.
Other GPU platforms and full-film temporal quality still require validation.

The Float16 path is checked against both upstream fixtures (0.12-0.14% relative
RMS, the same as before), every generated tiling against a CPU convolution, and
RGB-matrix against planar inputs (bit-identical). Its kernels are deterministic.

Device images, per-tile activations and network weights are retained by the
denoiser under its lock, then freed by `release!`. Public results own their
memory and are not overwritten by later calls. The source HDR cache remains
immutable.

The Float32 graph path keeps its fused bias/ReLU/Float16 write-back on Vulkan
(`fuseconv=false` is the A/B control); small atomic split-K layers there are not
bit-reproducible.

### Measured performance (Radeon 8060S, 1920×1080)

The same cached ALMA HDR frame and official large models, both auxiliary
prefilters, warmed-up full-pipeline latency including host transfers:

| Implementation | Median |
| --- | ---: |
| Original native Float32 / 768-pixel tiles | 2.448 s |
| DNNKernels graph, fp16, fused write-back (Vulkan) | 0.818 s |
| DNNKernels graph, fp16 (ROCm) | 0.736 s |
| **HWC port, Vulkan** | **0.178 s** |
| **HWC port, ROCm** | **0.146 s** |
| Released OIDN 2.5.1 HIP, same GPU, same session | 0.163 s |

On ROCm the port is 10% faster than upstream. The Vulkan gap was traced with
`BlackholeImage/tools/oidn_isa_probe.jl` (RADV assembly through
`VK_KHR_pipeline_executable_properties` beside `code_gcn`), and **it is not
Lava's SPIR-V**: the same kernel hand-written in GLSL and compiled by glslang
gets the same RADV code and runs slower (6.22 against Lava's 5.78 ms on the
64 -> 64 layer, bit-identical output; `BlackholeImage/tools/glsl/conv64.comp`).
What was fixed in the kernel source for RADV/ACO:

- branch-free gather (a guarded load serialised the prefetch on `vmcnt(0)`);
- zero padding selected where the value is staged, not after the load;
- unconditional prefetch on Lava (`prefetch`): under `if`, the prefetched
  registers were phis that ACO resolved with copies, i.e. waits. LLVM coalesces
  them, so ROCm keeps the branch (0.144 s against 0.157 s the other way);
- 16-byte staging through `uvec4` (Lava's `loadwu4` cooperative-matrix load);
- unsigned index arithmetic, and `!range` metadata on Lava's builtins.

What remains is RADV's lowering of a cooperative-matrix load from shared
memory: element-wise loads that its vectorizer merges only to 8 bytes, plus about
50 register moves per K-step to assemble WMMA operands. On RDNA3 the WMMA runs on
the vector ALU, so those moves compete with it: 141 non-WMMA VALU instructions
per K-step against 93 on ROCm. That needs a Mesa change
(`radv_nir_lower_cooperative_matrix.c`). Two other differences were measured and
are not limiting: 64-bit address arithmetic (RADV's scalar-base addressing via
32-bit offsets changed nothing), and ACO's register policy (Vulkan-specific
tilings for occupancy changed nothing). The Vulkan figure also needs the Mantle
readback fix of 2026-10-12 (25 MB download: 71 ms uncached, now 1.4 ms).

The 1080p output differs from upstream HIP by 0.204% relative RMS and from
upstream's fp32 CPU result by 0.142%, closer than upstream HIP itself is
(0.186%). These are not cross-platform guarantees; other GPUs need their own
`CONV_TABLE` sweep.

Reproduction tools in the film project: `tools/benchmark_oidn_native.jl` and
`tools/benchmark_oidn_reference.py`. No raytraced source frames are modified.
