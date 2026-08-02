# portability — report

Append, newest last. One entry per working session: what was done, what was
measured, what was **dis**proved. Negative results in full — roughly half the
value in `plans/perf-plan.md` is knowing which things were tried and lost.

Numbers only from the machine this project is assigned to, and engine
comparisons only from the desktop (`plans/GUARDRAILS.md` §6).

---

## 2026-08-02 — Phase 1: first execution on RDNA 3.5

### Baseline

| repo | branch | commit |
|---|---|---|
| `dev/Lava` | `sd/nvidia` | `e5e17ef82496d79d7d801a0babf1ae8cc30ebd27` |
| `dev/JuliaVision` | `sd/portability` (off `main`) | `c3512c46828dc0d67344ff695f075dfc30336571` |

Machine: AMD Ryzen AI MAX, Radeon 8060S (RADV STRIX_HALO), RDNA 3.5, 32 GB
unified. Mesa 26.1.5, Vulkan 1.4.354, Julia 1.12.6.

Environment is the workspace root, with `dev/Lava`,
`dev/JuliaVision/DNNKernels` and `dev/JuliaVision/GPUFiltering` developed into
it. Note for anyone reproducing: Lava fails to *precompile* headless, because
GLFW's `__init__` requires `DISPLAY`. That is not a device-creation failure and
not a port problem; it happens before any Vulkan call. `DISPLAY`, `XAUTHORITY`
and `XDG_RUNTIME_DIR` must be set in the environment that runs precompilation,
not merely in the one that runs the tests.

### Device capability dump

Everything below is queried from this device, not recalled.

```
device                              AMD Radeon 8060S Graphics (RADV STRIX_HALO)
driver                              Mesa 26.1.5 (RADV), Vulkan 1.4.354
subgroup_size                       64
subgroup size control               min 32, max 64, compute may pin: true
max_compute_shared_memory_size      65536
max_compute_work_group_invocations  1024
max_compute_work_group_size         (1024, 1024, 1024)
max_compute_work_group_count        (4294967295, 65535, 65535)
Lava.WORKGROUP_LIMIT[]              1024
Lava.shader_core_count()            40
Lava.shader_warps_per_sm()          nothing        <- see finding 3
Lava.max_shared_memory()            65536
coopmat_available                   true
coopmat2                            all eight false
coopvec_available                   false
ray_query_available                 true
ser_available                       false
maximal_reconvergence_available     true
subgroup_uniform_control_flow       true
subgroup_rotate_available           true
video_decode_available              true
```

Subgroup supported operations: `BASIC | VOTE | ARITHMETIC | BALLOT | SHUFFLE |
SHUFFLE_RELATIVE | CLUSTERED | QUAD | ROTATE | ROTATE_CLUSTERED`, across all
graphics, compute, ray-tracing, task and mesh stages, with
`quad_operations_in_all_stages = true`.

`VK_AMD_shader_core_properties` (v1), which Lava does not currently query:

```
shader_engine_count 2   shader_arrays_per_engine_count 2
compute_units_per_shader_array 10   -> 40 CUs, matching activeComputeUnitCount
simd_per_compute_unit 2   wavefronts_per_simd 16   -> 32 waves per CU
wavefront_size 64
vgprs_per_simd 768   max_vgpr_allocation 256   min_vgpr_allocation 12
sgprs_per_simd 1728  max_sgpr_allocation 108
```

#### Cooperative matrix shapes, decoded

The driver reports **14** `VkCooperativeMatrixPropertiesKHR` entries. All are
`16x16x16` at `SCOPE_SUBGROUP`. There is no other extent on this device.

| # | M,N,K | A | B | C | Result | saturating |
|---|---|---|---|---|---|---|
| 1 | 16,16,16 | UInt8 | UInt8 | UInt32 | UInt32 | false |
| 2 | 16,16,16 | UInt8 | UInt8 | Int32 | Int32 | false |
| 3 | 16,16,16 | UInt8 | UInt8 | Int32 | Int32 | true |
| 4 | 16,16,16 | UInt8 | Int8 | UInt32 | UInt32 | false |
| 5 | 16,16,16 | UInt8 | Int8 | Int32 | Int32 | false |
| 6 | 16,16,16 | UInt8 | Int8 | Int32 | Int32 | true |
| 7 | 16,16,16 | Int8 | UInt8 | UInt32 | UInt32 | false |
| 8 | 16,16,16 | Int8 | UInt8 | Int32 | Int32 | false |
| 9 | 16,16,16 | Int8 | UInt8 | Int32 | Int32 | true |
| 10 | 16,16,16 | Int8 | Int8 | UInt32 | UInt32 | false |
| 11 | 16,16,16 | Int8 | Int8 | Int32 | Int32 | false |
| 12 | 16,16,16 | Int8 | Int8 | Int32 | Int32 | true |
| 13 | 16,16,16 | Float16 | Float16 | Float16 | Float16 | false |
| 14 | 16,16,16 | Float16 | Float16 | Float32 | Float32 | false |

**`Float16` is the only float operand type. There is no `Float32` A/B form.**
`GEMM_TILE = 16` happens to be the only tile this device can do, so it is
correct here by coincidence rather than by agreement.

`test_coopmat_shape.jl`'s docstring says this device "lists 16x16x16 four ways".
It lists it fourteen ways, which collapse to six distinct
`(ab_type, c_type)` pairs. The docstring's point stands and its four examples are
all real; only the count is low.

#### Shared memory vs residency, measured

Atomic entry/exit probe, not a derivation. Each workgroup is 64 lanes (one
wave64); lane 1 atomically increments a live counter on entry and decrements on
exit, and records the counter value it observed. 8192 workgroups launched per
point, peak taken over all of them. Every lane strides the whole `@localmem`
array and lane 1 reads its far end, so the allocation cannot be optimised away.

Spin length was calibrated first, at 256 B: peak was **1280 at every one of
spin = 2 000, 10 000, 50 000 and 200 000**, a 100x range, so the number is a
plateau and not an artefact of how long groups happen to live. The table below
uses spin = 10 000.

| LDS per group | peak concurrent wg | waves per CU | groups run | live counter after |
|---|---|---|---|---|
| 256 B | 1280 | 32.0 | 8192 | 0 |
| 1024 B | 1280 | 32.0 | 8192 | 0 |
| 4096 B | 640 | 16.0 | 8192 | 0 |
| 8192 B | 320 | 8.0 | 8192 | 0 |
| 16384 B | 160 | 4.0 | 8192 | 0 |
| 32768 B | 80 | 2.0 | 8192 | 0 |
| 49152 B | 40 | 1.0 | 8192 | 0 |
| 65536 B | 40 | 1.0 | 8192 | 0 |

`groups run` is full at every point (GUARDRAILS §3: a kernel that skips work
looks like a speed-up), and the live counter returns to zero every time.

The measurements fit `waves per CU = min(32, 65536 / bytes)` exactly at all eight
points, so **LDS per CU is 64 KB and the wave cap is 32 per CU, over 40 CUs**.

That 32 is the same number as `wavefronts_per_simd x simd_per_compute_unit`
= 16 x 2 from the AMD v1 properties above. The measurement and the query agree
independently, which is what makes finding 3 actionable rather than speculative.

### Pass / fail / skip

| test | result | note |
|---|---|---|
| `test_subgroup_shuffle.jl` | **520 / 520 pass** | at subgroup width 64 |
| `test_coopmat_shape.jl` | **46 / 46 pass** | |
| `test_coopmat_perelement.jl` | **2 / 2 pass** | coopmat2 half skipped cleanly, as intended |
| `test_workgroup_limit.jl` | **51 / 51 pass** | `WORKGROUP_LIMIT` 1024 holds here |

No skip in this set was a run that should have happened. The one skip,
`VK_NV_cooperative_matrix2` per-element ops in `test_coopmat_perelement.jl`, is
correct: `ctx.coopmat2` is eight `false`s on this device.

### The three unverified claims: all three hold on AMD

All three shipped in docstrings on 2026-08-02 on the strength of "it is KHR, so
it should work", and none had executed on AMD. Executed now:

1. **`coopmat_mul`** (`OpFMul` on two cooperative matrices, component-wise).
   Verified. `fmul_rowscale!` produces the exact expected product for both
   offsets.
2. **The stride-0 broadcast load.** Verified, in the same kernel: a stride-0
   `AcceleratedMatrix` broadcasts 16 row factors across 16 columns and one
   `OpFMul` applies them.
3. **The subgroup shuffle family and `subgroup_rotate`.** Verified, 520 asserts,
   at width **64**. `shuffle`, `shuffle_xor`, `shuffle_up`, `shuffle_down`,
   `broadcast` and `rotate` all behave; the butterfly reduction built on
   `shuffle_xor` agrees exactly with `subgroup_add` and with the host; and the
   round-trip holds for `Float32`, `Float64`, `Int32`, `UInt32`, `Int64`,
   `UInt64`.

`test_workgroup_limit.jl` also confirms the fourth thing the brief asked about:
the 256-lane cap really was our pipeline-cache hash collision and not a device
limit. 1024-lane workgroups run every lane they are given here too, so that
conclusion generalises off the card it was found on.

### Findings

#### 1. Subgroup width is not a device property on this hardware — design impact

`subgroup_size` is 64, against 32 on the desktop. That much was anticipated. What
was not: `VkPhysicalDeviceSubgroupSizeControlProperties` reports
**min 32, max 64, and compute pipelines may pin either**. RDNA 3.5 compiles
compute at 32 *or* 64 depending on how the driver chose, and a pipeline can
require one.

So `Lava.device_subgroup_size()` returns a **default**, not a fact about the
kernel that is about to run. Any plan object that stores a width without also
pinning it via `VK_EXT_subgroup_size_control` at pipeline creation is storing a
guess that the driver is free to contradict.

This matters directly to `kernels-refactor` step 3, which is about to define plan
objects, and it is invisible on the desktop, where 32 is the only answer the card
gives. Concretely: a plan needs a width **field**, that field must be pinned at
pipeline creation, and `flash.jl`'s five literal `32`s
(`flash.jl:399, 431, 440` and the `NT = NW * 32` / `tid ÷ 32` pair) are wrong
here in a way that no amount of care on one card would reveal.
`Lava.can_require_subgroup_size(ctx, n)` already exists and answers whether a pin
is legal.

#### 2. `test_coopmat_perelement.jl` hardcoded 32 lanes — FIXED (test-level)

The file said *"The subgroup is 32 lanes here; a workgroup smaller than one has
undefined cooperative-matrix behaviour, so the launches below are 32 wide"* and
launched every kernel at `(32,)`. On this device the subgroup is 64, so those
launches were **half a subgroup**, which is exactly the undefined-behaviour case
the comment itself names.

It passed anyway, which is the point: it was getting a correct answer out of
undefined behaviour, and would have kept doing so until it silently did not.

Fixed at test level, which the brief permits: the launches now ask
`Lava.device_subgroup_size(ctx)`. Re-run at the correct width 64: still 2/2, so
the KHR claims above are verified rather than lucky.

#### 3. `shader_warps_per_sm()` returns `nothing` on AMD, but the number exists — FILED

`query_device_compute` (`Lava/src/runtime/device.jl:285`) branches NVIDIA to
`VK_NV_shader_sm_builtins`, else AMD to `VK_AMD_shader_core_properties`**2**.
Properties2 carries only `shaderCoreFeatures` and `activeComputeUnitCount`, so
the AMD branch fills `sm_count` and leaves `warps_per_sm` at 0, which
`shader_warps_per_sm()` maps to `nothing`.

The wave counts are in `VK_AMD_shader_core_properties` **v1**, which this device
also advertises and Lava never opens:
`wavefronts_per_simd (16) x simd_per_compute_unit (2)` = **32 waves per CU**.

That figure is independently confirmed by the occupancy probe above: measured
peak 1280 over 40 CUs is 32 per CU. So the fix is a query, not a guess.

The docstring at `device.jl:250` ("AMD through
`VK_AMD_shader_core_properties2`") describes only the half that yields the CU
count and should mention v1 for the wave count.

**Filed, not patched** — `device.jl` is being rewritten by `lava-core` phase 2.

Note also that no vendor-neutral query exists: core Vulkan and KHR have nothing
for occupancy, which is why llama.cpp does the same vendor split at
`ggml-vulkan.cpp:6138`. Intel, Apple and lavapipe would still return `nothing`
after this fix. The portable answer is the probe in this report, which also has
the advantage of measuring what the kernel actually gets rather than what the
hardware could hold.

#### 4. A 48 KB shared-memory budget wastes 16 KB here at zero occupancy cost

`max_shared_memory()` is **65536** here against 49152 on the desktop, so
`FLASH_SHARED_BUDGET = Ref(48 * 1024)` (`flash.jl:112`) is a module-level
constant that is simply wrong on this device.

The occupancy table makes the size of the mistake precise, and it is larger than
"33% less memory": **49152 B and 65536 B both give exactly 1 workgroup per CU**.
Going to the real limit costs nothing in residency. A kernel tuned to 48 KB on
NVIDIA leaves a third of this device's LDS unused for free.

Review finding 9a already flags this pair as module-level `const`s that cannot
simply call a device query, since that would build a Vulkan context during
precompilation. The number is per-device, so the query has to move to the use
site.

#### 5. `Bool` capability predicates cannot express this device — evidence for finding 1

`Lava.coopmat_gemm_available()` is `true` here. But the shape table has no
`Float32` A/B form at all, so a Float32 GEMM asks the predicate, is told yes, and
emits cooperative-matrix instructions this device does not implement.

This is the same failure `test_coopmat_shape.jl` was written to catch one level
up (extent match ignoring operand type), and it is review finding 1's argument
made concrete on real hardware: a capability is a *set of shapes and types*, and
collapsing it to a `Bool` throws away exactly the part that differs between
vendors.

#### 6. `coopmat_shapes` keeps `a_type` only, discarding `b_type` — FILED, low severity

`device.jl:1155` records `ab_type = UInt32(p.a_type)`, so the 14 driver entries
collapse to 6 distinct tuples. Four of the 14 have `a_type != b_type` (mixed-sign
int8), and those are recorded as though both operands had `a_type`.

No false positive arises **on this device**, because every mixed form is
accompanied by its homogeneous form. But the shape is unsound: a device
advertising only `A=UInt8, B=Int8` would cause `coopmat_shape(ctx, UInt8, ...)`
to answer `true`, and Lava would then emit a matrix pair the device never
advertised. `saturating_accumulation` is dropped the same way.

Low severity, filed rather than patched, and worth folding into whatever
capability type replaces the predicate (finding 5 above).

### Environment note, not a port issue

`Abacus` fails to precompile in this workspace on
`UndefVarError: reset_runtime not defined in GPUCompiler`. Unrelated package,
unrelated to Lava or DNNKernels, recorded only so the next person does not chase
it.
