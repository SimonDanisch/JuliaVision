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

#### 7. `video_capabilities` omits a required pNext chain and segfaults RADV — FILED, patch below

**This is a hard crash that killed the whole Lava suite.** Rule 0 applies and
comes out on our side: the call is invalid per spec.

`vkGetPhysicalDeviceVideoCapabilitiesKHR` requires, for an H.264 decode profile,
that `pCapabilities->pNext` contain **both** `VkVideoDecodeCapabilitiesKHR` and
the codec-specific `VkVideoDecodeH264CapabilitiesKHR`
(VUID-vkGetPhysicalDeviceVideoCapabilitiesKHR-pVideoProfile-07185 and -07183).

`video.jl` builds that chain correctly in **two** places
(`decode_capability_flags` at :304, and the `H264Decoder` constructor at :594)
and passes **`C_NULL`** in the third, `video_capabilities` at :337. RADV writes
the codec-specific capabilities unconditionally, through a chain entry that is
not there, and segfaults inside `libvulkan_radeon.so`.

Isolated to a two-call MWE on this device, which is what makes it ours rather
than a driver defect: same function, same profile, same process.

```
A: decode_capability_flags  (full chain)   -> returns 0x00000002, fine
B: video_capabilities       (pNext=C_NULL) -> SIGSEGV in libvulkan_radeon.so
```

With the chain added, B returns real data:
`minBitstreamBufferOffsetAlignment = 128`, `minBitstreamBufferSizeAlignment = 128`,
`maxDpbSlots = 17`, `maxActiveReferencePictures = 16`.

Two side notes from that number. `test_video_decode_layout.jl`'s header says
"The H.264 profile on AMD asks for 4096"; on this driver (Mesa 26.1.5) it asks
**128**. And `decode_capability_flags` returns `0x2`, DISTINCT only, so this
device is exactly the DISTINCT-only configuration that test says it exists to
pin, and it crashed before reaching that assertion.

Reach is narrow: `video_capabilities` has exactly one caller outside its own
definition, the test at `test_video_decode_layout.jl:42`. The decoder itself is
unaffected because :594 chains correctly. But it is a documented public function
that hard-crashes, and it takes the suite with it.

The fix is to make :337 look like :304, which is twenty lines above it.

#### 8. A refused pipeline creation returns a NULL handle that Lava binds — FILED, patch below

The second hard crash, at `vkCmdBindPipeline`, from inside
`no_pipeline_compilation`. Also ours, and **not vendor-specific: it is
Linux-specific, so it is equally live on the desktop.**

`VK_PIPELINE_COMPILE_REQUIRED` is a **success**-class code (positive,
1000297000). Vulkan.jl's generated `_create_compute_pipelines` therefore does not
raise: `@check` only converts the negative error class, and it returns
`(pipelines, _return_code)` with `pipelines[1] == VK_NULL_HANDLE`.

`pipeline.jl:518` then writes

```julia
pipelines, _ = @vk_checked "vkCreateComputePipelines" Vulkan.create_compute_pipelines(dev, [ci]; pipeline_cache)
```

and **discards the return code**, which is the only thing that says the handle is
null. `@vk_checked` is `unwrap(...)`, which cannot help here for the same reason.
The null pipeline is cached, bound later, and the driver dereferences it.

The check for exactly this code **does** exist, at `pipeline.jl:504` — but inside
`if LARGE_STACK_PIPELINE`, and `LARGE_STACK_PIPELINE = Sys.iswindows()`. So the
refusal is detected on Windows and silently mishandled on every Linux machine.
The constant is even defined unconditionally at :137 with a comment noting that
it is success-class and "needs naming".

Consequences beyond the crash, and these are the reason this matters more than
one test:

- `PIPELINE_COMPILES_REFUSED[]` is only incremented on the Windows branch, so it
  is always 0 on Linux.
- `PIPELINE_COMPILE_MISSES` is populated from the `catch` at :401, which matches
  on the string `"PIPELINE_COMPILE_REQUIRED"` in a raised exception. Nothing
  raises that on Linux, so the list never fills.
- So `no_pipeline_compilation`, the instrument whose entire purpose is to prove
  the pipeline cache is doing its job, **cannot report a miss on Linux**. Any
  Linux run of it either crashes or returns a false green. That affects the
  frozen-kernel-cache "verify with zero misses" workflow directly.

Patch applied locally to get the rest of the suite to run, **not committed**:
capture the code and raise on it in the non-Windows branch too, mirroring :504.

#### 9. The pipeline-cache test's negative control does not fire on RADV

With finding 8 patched the test no longer crashes, but it fails 2 of 9:

```
test_pipeline_cache_no_compile.jl:62  @test refused                        FAIL
test_pipeline_cache_no_compile.jl:63  @test PIPELINE_COMPILES_REFUSED == 1 FAIL
```

The device supports `pipelineCreationCacheControl` and Lava does enable it
(`device.jl:1023`), so the flag is legitimate. RADV nonetheless creates the
"novel" pipeline rather than returning `VK_PIPELINE_COMPILE_REQUIRED`.

Tried and **did not** fix it: deleting Lava's on-disk `VkPipelineCache` blob, and
running with `MESA_SHADER_CACHE_DISABLE=true` (the hypothesis being that Mesa's
own shader cache, which is independent of `VkPipelineCache`, was satisfying the
request). Both still fail the control.

That is significant because the test file says so itself: *"The negative control
is not optional. A green run against an instrument that never fires proves
nothing."* On this device the instrument does not fire, so the rest of that
testset is vacuous here, and the same is true of anything else built on
`no_pipeline_compilation`. Not root-caused; handing over as an open question.

#### 10. `vkUnmapMemory` in the buffer free path segfaults — LOCATED, patch below

**Correction to my own first reading of this.** I initially recorded it as a
teardown crash because that is where it first appeared. It is not. Once findings
7 and 8 were patched and the suite got further, the same stack fired **in the
middle of `test_static_workgroup.jl`**, from a finalizer that GC happened to run
at that moment. So this can kill any Lava process at an arbitrary GC point, not
only at exit. That makes it the most disruptive of the three.

```
destroy_buffer! (memory.jl:858) -> unmap_memory -> vkUnmapMemory -> SIGSEGV
  <- vk_free! (memory.jl:821) <- lavaarray.jl:73 <- GPUArrays release
  <- unsafe_free! (lavaarray.jl:238) <- run_finalizers
```

The code at `memory.jl:856` wraps the unmap in `try`/`catch` with the comment
*"unmap may fail if the driver released the memory first ... don't propagate
(finalizers must not throw)"*. **That protection cannot work.** An invalid unmap
is undefined behaviour, not a Julia exception:
VUID-vkUnmapMemory-memory-00689 requires the memory to be currently mapped, and a
SIGSEGV is not catchable. The `catch` gives the appearance of having handled the
case while handling nothing.

The call is also **redundant**: `vkFreeMemory` implicitly unmaps, and
`buf.memory.destructor()` runs eleven lines below. Deleting the unmap removes the
crash entirely: `test_static_workgroup.jl` goes from SIGSEGV to completing with
541 passed.

Patch applied locally, **not committed**, and deliberately labelled an experiment
rather than a proposed fix, because removing the call treats the symptom. The
reason the unmap faulted is that `buf.mapped_ptr` and the memory's real mapped
state disagree, and *that* disagreement is the actual defect. Removing a
redundant call is safe and spec-legal; it is not a diagnosis.

Two earlier hypotheses, **both disproved by MWE** and recorded so nobody repeats
them:

1. *Plain `LavaArray` finalization at exit is broken.* No: allocate, use, and
   either hold to exit or free explicitly. Three modes, all exit 0.
2. *`no_pipeline_compilation`'s `empty!(PIPELINE_CACHE)` drops pipelines that
   `LAUNCH_PLAN_CACHE` / `LINKED_KERNEL_CACHE` still reference, so a finalizer
   destroys a live `VkPipeline`* (the GUARDRAILS §8 class). No: clear the cache,
   `GC.gc(true)`, relaunch the same kernel. Clean, exit 0.

Operational note worth keeping regardless: this crash block-buffers away the test
output that preceded it, which is why the first suite log was 3 KB of pure stack
trace. **Run Lava suites on this machine under a pty** (`script -qec "julia ..."`)
so output survives.

#### 11. The rank>=3 `Extruded` miscompile does NOT reproduce on RDNA 3.5 — this is `lava-core` Phase 1's missing datapoint

The most useful thing this machine found, and it was hidden behind finding 10.

`test_static_workgroup.jl` completes with **541 passed, 11 failed**. Every one of
the 11 fails in the same direction:

```
:133  coverage(ND, wg, :typed; fallback=false) < 1.0     Evaluated: 1.0 < 1.0   (x4)
:143  coverage(ND5, (16,4,1,1,1), :typed)      < 1.0     Evaluated: 1.0 < 1.0
:144  coverage(ND5, (16,4,2,1,1), :typed)      < 1.0     Evaluated: 1.0 < 1.0
:155  law(b2, b3) == min(1, b3/b2)                       Evaluated: 1.0 == 0.5
:155  law(b2, b3) == min(1, b3/b2)                       Evaluated: 1.0 == 0.5
:155  law(b2, b3) == min(1, b3/b2)                       Evaluated: 1.0 == 0.125
:155  law(b2, b3) == min(1, b3/b2)                       Evaluated: 1.0 == 0.25
:155  law(b2, b3) == min(1, b3/b2)                       Evaluated: 1.0 == 0.125
```

These are **characterization tests that assert the defect is present** (`< 1.0`,
and a coverage law of exactly `min(1, b3/b2)`). They fail here because coverage is
**1.0 everywhere**: the same Lava-emitted SPIR-V that writes only `b3/b2` of its
output on the desktop writes **all** of it on RDNA 3.5.

`lava-core`'s brief lists this as one of two items labelled *NVIDIA driver
miscompile*, root-caused, mitigated, and "cannot be settled on this hardware".
This is the second device it could not get. What the result means, stated per
Rule 0's own corollary rather than more loosely:

> when our module behaves *differently* on another vendor, that difference is
> evidence about **our** code, not evidence against the driver.

So this does **not** exonerate our emitter, and it is not a licence to call it a
driver bug. It says our module's correctness depends on something the spec leaves
open, which one compiler exploits and the other does not. That is exactly the
shape of the lead already in the brief: *"LLVM's canonicalisation of `x <s 1` when
it believes `x` is non-negative, valid only under `nuw`/`nsw` flags that rank 4's
extra index arithmetic supplies."* RADV's compiler evidently does not take the
same liberty. The GLSL differential (Rule 0 instrument 3) is now much cheaper to
interpret, because there is a known-good vendor to diff against.

**Separately, the test itself needs changing, and by this project's own rule.**
`CLAUDE.md` says no vendor-conditional tests: "the regression test pins the source
pattern and runs the same assertion on every platform." These assertions pin the
**symptom**, so they can only pass on hardware that exhibits the defect, and they
will keep failing on every non-NVIDIA device until the underlying bug is fixed, at
which point they fail everywhere. They should assert full coverage on every
device, with the NVIDIA shortfall recorded as a known failure, not as the expected
result.

#### 12. Lazy artifacts never downloaded — FIXED (not platform-specific)

`DNNKernels.artifactpath` (`assets.jl:78`) called
`Artifacts.ensure_artifact_installed`. That binding does not exist in the
`Artifacts` stdlib; it is in `Pkg.Artifacts` / `LazyArtifacts`. Every lazy fetch
therefore raised `UndefVarError`, the surrounding `catch` downgraded it to a
`@warn`, and `assetpath` fell through to its "local generated tree" branch, which
in a checkout with no `gen/` is `findasset` walking to the filesystem root and
returning `/gen/graphs/sam2-large`.

**Nothing to do with AMD.** It fails identically everywhere and stayed hidden
only because the machines that run these models have a local `gen/` and never
took the artifact path. `assets.jl` already declares `LazyArtifacts` in
`Project.toml` and imports it at line 32 — the dependency was added for this call
and the call was then written against the wrong module.

Fixed (one word). With it, `SAM2Runner.assetdir()` fetches the 942 MB
`sam2-large` artifact and `sam2model()` builds in **39.9 s** on this device.

Worth noting as a pattern rather than an incident: the `catch` is what allowed a
plain typo to survive indefinitely, by turning a hard failure into a plausible
fallback path. It is the shape `CLAUDE.md` warns about.

#### 13. Correcting my own earlier claim: none of this needs CUDA

I asserted mid-session that the DNNKernels suite cannot run here because its
reference activations require a CUDA export. **That was wrong**, and it is worth
recording because it would have written off a whole verification path.

Both asset sets are ordinary lazy downloads: `sam2-large` (942 MB, weights and
the two graph JSONs) and `sam2-large-refs` (1.2 GB, the PyTorch reference
activations, deliberately separate because a caller that just wants to segment a
picture should not fetch them). No export, and therefore no CUDA, is involved in
either running or verifying SAM 2 on this machine.

The real blocker is much narrower and is a wiring gap:
`DNNKernels/test/runtests.jl:37` resolves its asset directory with
`findasset("gen")`, which only walks parent directories, whereas
`SAM2Runner.assetdir()` uses `assetpath(...)`, which consults `Artifacts.toml`.
So the suite cannot see an artifact it has every right to use. That is a
one-line change and it makes the layer-by-layer PyTorch comparison runnable on
any machine, which is exactly the cross-vendor numerical check this project
wants. Filed rather than changed here, since it belongs with whoever owns the
verification story.

### Design note for `kernels-refactor` step 4 and `lava-core` phase 3

Written here rather than implemented, because the brief says gather evidence and
do not write per-feature fallbacks yet, and because the dispatch API belongs on
`sd/kernels-refactor`. What this machine adds is that both projects are designing
against one card's answers, and two of those answers are wrong here.

**Rule for the whole design: types are named after capability levels, never after
vendors.** `CoopMatMapped`, not `CoopMatNV`. A card that gains per-element
operations next year should acquire an existing type, not need a new branch. This
is the same rule as Lava's own ban on vendor-conditional code in `src/`, and it is
what makes "code using Lava never writes something for one vendor" enforceable
rather than aspirational.

#### Vendor differences come in three kinds, and they need different treatment

Collapsing them is what produces `if has_feature` chains.

1. **Same instruction, different width.** The subgroup shuffle family: verified
   identical here at 64 lanes and on the desktop at 32. The *intrinsic* is
   already device-independent. What leaks is `subgroup_size()`, and the fix is
   that no kernel writes 32 (see finding 1).
2. **Instruction exists on one vendor only.** coopmat2 per-element and reduce,
   cooperative vector. There is no device-independent intrinsic to write here.
   There is a device-independent *operation* with two lowerings.
3. **Instruction is KHR, but this device's shape and type table does not include
   your case.** `Float32` cooperative-matrix operands here. This is the kind a
   `Bool` predicate handles worst, because the answer is "yes, but not for you"
   (finding 5).

#### Capability as a type, carrying shapes rather than a flag

```julia
abstract type CoopMat end
struct NoCoopMat            <: CoopMat end
struct CoopMatBasic{M,N,K}  <: CoopMat end   # KHR: load/store/muladd/getcomp/setcomp
struct CoopMatMapped{M,N,K} <: CoopMat end   # + per-element and reduce

coopmat(ctx, ::Type{AB}, ::Type{Acc}) -> CoopMat
```

On this device `coopmat(ctx, Float16, Float32)` is `CoopMatBasic{16,16,16}()` and
`coopmat(ctx, Float32, Float32)` is `NoCoopMat()`. On the desktop the first would
be `CoopMatMapped{16,16,16}()`. `matmul!` then dispatches, and a new path is a new
method rather than another `&&` in a predicate:

```julia
matmul!(out, A, B) = matmul!(coopmat(ctx, eltype(A), eltype(out)), out, A, B)
matmul!(::NoCoopMat,    out, A, B) = ...
matmul!(::CoopMatBasic, out, A, B) = ...
matmul!(::CoopMatMapped,out, A, B) = ...
```

#### The coopmat2 fallbacks are constructible from KHR, not hypothetical

This matters because it means kind 2 above collapses into kind 3 for the two
operations that block `kernels-to-port.md` item 17. `coopmat_getcomp` and
`coopmat_setcomp` are wired KHR and work here, so:

- `coopmat_map(f, m)` lowers to `OpCooperativeMatrixPerElementOpNV` on
  `CoopMatMapped` and to a `getcomp`/`setcomp` loop on `CoopMatBasic`.
- `coopmat_reduce(op, m)` lowers to `OpCooperativeMatrixReduceNV` on
  `CoopMatMapped` and to `getcomp` plus a subgroup butterfly on `CoopMatBasic`.

One operation, two lowerings, chosen by dispatch, no capability test at any call
site. The butterfly is exactly where finding 1 bites: it must read the plan's
pinned width, not a literal 32. `test_subgroup_shuffle.jl` passing 520/520 at
width 64 here is what makes that fallback safe to write.

Not written now: per the brief, and because `lava-core` phase 3 already schedules
each instruction to land "with its KHR fallback as a sibling method", which is
only cheap once plan dispatch exists.

### Environment note, not a port issue

`Abacus` fails to precompile in this workspace on
`UndefVarError: reset_runtime not defined in GPUCompiler`. Unrelated package,
unrelated to Lava or DNNKernels, recorded only so the next person does not chase
it.

### SAM 2 (large) actually runs on this device

Asked directly, so measured directly. Synthetic input (a centred bright disc,
centre click) rather than the reference activations, because `refs.safetensors`
is in the separate 1.2 GB artifact and encoder cost does not depend on pixel
values.

```
model build            39.8 s
cold run                0.488 s     frozen cache 106 hits / 0 misses
steady state (5 runs)   min 0.2899 s   median 0.2941 s
desktop reference       encode 100.4 ms + decode 3.30 ms ~= 104 ms
ratio                   ~2.8x slower than the RTX 4000 Ada
```

**It is correct, not merely non-crashing.** The mask's positive fraction is
**0.192** against a disc covering **0.196** of the frame, and all 65536 logits
are finite (range -8.578 to 4.370). It found the object.

For a 40-CU integrated part sharing system memory, 2.8x off a discrete Ada card
is respectable, and it is a **floor**: this ran with `coopmat2` entirely absent
(so the per-element flash path falls back), `FLASH_SHARED_BUDGET` at 48 KB
against a measured 64 KB at identical occupancy (finding 4), and
`FLASHCM_MINGRID` at 48 against 40 real CUs.

Two honest qualifications:

- **The frozen-cache number is weaker than it reads.** `misses = 0` does show the
  freeze/replay mechanism works on AMD: every kernel the workload touches was
  captured and replayed from disk with no fallback compile. It is **not** evidence
  that NVIDIA-recorded SPIR-V runs here. `~/.julia/scratchspaces/lava_frozen_kernels`
  is a local scratchspace and was written during this machine's own precompile
  workload. Shipping the cache is what would test the stronger claim.
- **Predicted IoU came back exactly `0.0000`** while the mask is visibly right.
  That may be genuine low confidence on a very out-of-distribution synthetic
  disc, or a fault in the score head. It cannot be distinguished without the
  reference activations. Flagged, not concluded.

Caveat on all of the above: it ran with three locally-applied Lava crash patches
(findings 7, 8, 10) and one known-unpatched crash site (`vkCmdCopyBuffer`,
`command.jl:1549`). The number is provisional until those land.

### Full suite results

#### Lava suite (with the three local crash patches)

Does **not** complete. Progressively deeper on each fix:

| run | patches applied | outcome |
|---|---|---|
| 1 | none | SIGSEGV in `video_capabilities` (finding 7), 3 KB of log, all prior output lost to buffering |
| 2 | video | SIGSEGV at `vkCmdBindPipeline` (finding 8) |
| 3 | video + pipeline | SIGSEGV at `vkUnmapMemory` (finding 10), mid-`test_static_workgroup` |
| 4 | all three | SIGSEGV at `vkCmdCopyBuffer` (`command.jl:1549`) in `test_closesthit_via_rayquery.jl:89` |

Run 4 reaches far more of the suite and reports **15 failures** before the crash,
which decompose cleanly:

- **13 in `test_static_workgroup.jl`** — finding 11, the characterization tests
  that assert the `Extruded` miscompile is present. They fail because it is not.
- **2 in `test_pipeline_cache_no_compile.jl`** — finding 9, the negative control
  that does not fire on RADV.

So there are **no unexplained Lava failures on this device**. Every one maps to a
recorded finding. The fourth crash site is open (task: root-cause the
`vkCmdCopyBuffer` use-after-free; it is plausibly the same buffer-lifetime class
as finding 10, since both are "we touched a buffer that is no longer valid").

#### DNNKernels

`runtests.jl` does not run here (finding 13, an asset-wiring gap, not hardware).
The six asset-independent files all run and are **entirely green: 325 passed, 0
failed**, including both coopmat-dependent ones:

| file | result |
|---|---|
| `test_constfold.jl` | 19 / 19 |
| `test_foldoutcasts.jl` | see finding 14 |
| `test_convtranspose_gemm.jl` | 37 + 19 / 56 |
| `test_transposeLE.jl` | 16 / 16 |
| `test_coopmat_attention.jl` | **10 / 10** |
| `test_flash.jl` | **213 + 11 / 224** |

`test_coopmat_attention.jl` and `test_flash.jl` passing matters: those are the
kernels that reach `Lava.coopmat_gemm!` and the flash paths, on a device with a
64-lane default subgroup and no coopmat2. The subgroup pin at `pipeline.jl:363`
is doing its job.

#### 14. Two testsets ran zero assertions and read as green — FIXED (test-level)

`test_foldoutcasts.jl` resolved its graph with a hardcoded
`../../../../gen/graphs/sam2-large`, and `test_constfold.jl` with the same. When
that path does not exist both guarded testsets took an `@info ... skipping`
branch with **no `@test_skip`**, so each reported `Total 0` — indistinguishable
from a pass in any summary.

This is precisely the bug `runtests.jl:36` documents as already fixed elsewhere:
*"Walked up from here rather than a fixed `../../../gen`: this package moved into
a monorepo and the fixed form silently pointed at `dev/gen`."* The fix never
reached these two files.

Measured, by pointing `gen/graphs/sam2-large` at the downloaded artifact:

```
before:  foldoutcasts 0    foldrelu 0          (Total 0, reads green)
after:   foldoutcasts 4023 foldrelu 149        (4172 assertions, all passing)
```

**4172 assertions were silently not running** on any machine without a local
`gen/` tree — which, until finding 12 was fixed, was any machine relying on
artifacts.

Fixed at test level: both files now use `DNNKernels.findasset`, and the skip
branches gained `@test_skip` so an absent graph shows as `1 Broken` rather than
`Total 0`. Verified in both directions.

## 2026-08-02, continued — validation turned on

The three crash patches are now committed to `dev/Lava` @ `sd/portability`
(`6dcf9f4`, `93e4765`, `bc61b3d`). With those in, the suite was re-run under
`LAVA_VALIDATION=1`, which is Rule 0's first instrument and which
`spirv-intrinsics.md` notes is "cheap, and neither is run by default".

#### 15. Lava emits SPIR-V that `spirv-val` REJECTS — two modules, both illegal

Not "depends on unspecified behaviour". **Invalid.**

```
vkCreateShaderModule(): pCreateInfo->pCode (spirv-val produced an error):
  Instruction may not have a logical pointer operand
    %127 = OpBitcast %_ptr_Workgroup__arr_float_uint_128 %126
    %99  = OpBitcast %_ptr_Workgroup__arr_float_uint_128 %98
```

Vulkan requires the **Logical** addressing model, under which `OpBitcast` may not
produce or consume a pointer at all. Two distinct modules do it, both reached
from the same construct: a clamped ternary over shared memory, which LLVM lowers
to an `OpSelect` between two Workgroup pointers.

- `test_select_width_mismatch.jl:22` — "OpSelect of Workgroup pointers (clamped ternary)"
- `test_shared_memory_stress.jl:242` — "iterative shared-memory stencil (clamped, 32 barriers, double-buffered)"

**Both of these tests PASS when validation is off.** They are absent from the
15 failures in the run without it. So the driver accepts an invalid module, the
kernel produces the right answer today, and nothing reports a problem — which is
the worst possible failure mode and exactly the argument for `lava-core` Phase
1's "turn `spirv-val` and `gpu_av` on by default for test runs".

Worth noting what this is *not*: `test_select_width_mismatch.jl` already exists
as a regression test for this construct. Its header describes the **previous**
bug — the emitter minting two structurally-identical-but-distinct `[N x T]` type
ids so `OpSelect`'s operand type differed from its result type. That was fixed by
making the ids agree. The `OpBitcast` on a logical pointer was left in place, and
it is independently illegal. The fix addressed the symptom the validator happened
to report first.

Reproduce:

    spirv-val <input.spv> --relax-block-layout --scalar-block-layout \
              --workgroup-scalar-block-layout --allow-localsizeid --target-env vulkan1.4

Nine test sites report an error under validation, but only these two distinct
spirv-val messages appear. The other seven (`test_atomics_and_dispatch.jl` x4,
`test_int32_cartesian_miscompile.jl` x2, `test_shared_index_division.jl`) are most
likely re-reporting accumulated messages rather than being independently invalid,
since `check_validation_errors!` reads a list that does not appear to be cleared
between tests. **Not claimed as seven separate bugs** — worth a look, because if
that list really is sticky then a validation error attributes itself to whichever
test runs next, which would be its own trap.

Deliberately NOT claimed: that this explains the `Extruded` miscompile (finding
11) or the `OpUDiv`-in-a-shared-store-index item. Both involve shared-memory
indexing and both are labelled driver miscompiles, so the adjacency is
suggestive and worth pursuing. But nothing here demonstrates a link, and Rule 0
cuts both ways: a satisfying-sounding connection is not evidence.

#### 16. The `vkCmdCopyBuffer` fault is a floating GC race, not a test-specific bug

Characterised, still open.

- `test_closesthit_via_rayquery.jl` **passes 3/3 standalone**. It only faults
  inside the full suite.
- Across runs the crash **moves**: `test_closesthit_via_rayquery.jl:89` in one
  run, `test_static_workgroup.jl:81` in the next. Always the same call,
  `Vulkan.cmd_copy_buffer` at `command.jl:1549`.
- Under the validation layer there is **no object-lifetime error** — no
  "destroyed VkBuffer", no invalid-handle VUID — and the layer is in the crash
  stack, having forwarded the call before RADV faulted. So the `VkBuffer` handle
  is still live as far as validation tracks it; the likelier shape is that its
  *memory* was freed while the buffer object remained valid, which validation
  does not track as aggressively.

This is the residual that `vk_free!`'s own comment predicts, at about the rate it
predicts: *"Not certainly the last of it ... either a second window exists or
something rarer shares this one. If it recurs, the next thing to check is whether
a buffer can be reached by an open batch through something `pins` does not count
either."*

One structural gap noticed while reading, **not confirmed on this hardware**: the
deferral check at `memory.jl:732` consults `ctx.default_bq` only. A buffer
reachable from a recording batch on a different `BatchQueue` would not be seen.
This device reports `max_queue_count = 1`, so it cannot be exercised here, but
`async_queue_count` is 4 and the RT/async paths are where a second queue appears.
Worth the desktop checking.

#### 15a. Reproducer for the invalid SPIR-V, and one negative result

No MWE needed — the existing regression test is already minimal, and the two
spellings differ only by an environment variable:

```
$ julia --project=. dev/Lava/test/test_select_width_mismatch.jl
  passes

$ LAVA_VALIDATION=1 julia --project=. dev/Lava/test/test_select_width_mismatch.jl
  Error During Test at test_select_width_mismatch.jl:22
    Instruction may not have a logical pointer operand
      %99 = OpBitcast %_ptr_Workgroup__arr_float_uint_128 %98
  0 passed, 0 failed, 1 errored
```

Seconds to run, one testset, no GPU dispatch needed past shader-module creation.

**Negative result, recorded so the next person does not repeat it.** I wrote a
hand-rolled MWE with the same shape — `@localmem Float32 (128,)`, the same
clamped ternary, `lava_local_invocation_index` instead of KA's `@index(Local)`,
compiled via `lava_compile_gpu` — and it **validates cleanly** (`spirv-val`
exit 0, 2012 bytes). So the illegal bitcast is not produced by the ternary alone;
it depends on the exact lowering the KernelAbstractions `@kernel` wrapper
produces. Anyone reducing this further should start from the KA path, not from a
plain function.

That is also why I stopped short of fixing it. The emitter has three pointer
`OpBitcast` sites (`emit.jl:1362`, `:1378`, `:1414`) and the observed result type
is a pointer-to-array, which matches none of the first two (both build
pointer-to-scalar from `map_type!(eff_load_ty)`). Without a reduction I can step
through, a fix here would be a guess at code `lava-core` owns. The bug is
localised, reproducible in one command, and has a named VUID; that is a better
handoff than a speculative patch.

Note `emit.jl:1317` already carries the comment *"would emit an identity
OpBitcast on a logical pointer, which Vulkan rejects
(VUID-StandaloneSpirv-Logical pointer-OpBitcast)"* — so the constraint is known
in the file, and guarded for the identity case only.

## 2026-08-02, final — two emitter/device bugs fixed, validation clean

Committed to `dev/Lava` @ `sd/portability`: `be52353`, `332dc33`, `046b1ed`.

#### 17. The invalid `OpBitcast` — FIXED

`emit.jl`'s pointer-select reconciliation already knew a PhysicalStorageBuffer
pointer cannot be `OpBitcast` and routed around it with a ptr→uint→ptr roundtrip.
Every **other** storage class fell through to the bitcast, which is equally
illegal: the Logical addressing model forbids `OpBitcast` on any pointer.

Drilling is the legal direction, since an aggregate pointer can reach its first
element with `OpAccessChain`. When the two select operands' pointees are
array-of-T and T in a logical storage class, make both element pointers:

```
%98  = OpAccessChain %_ptr_Workgroup_float %shmem %uint_0 %97      ; &sh[lid-1]
%99  = OpAccessChain %_ptr_Workgroup_float %shmem %uint_0 %uint_0  ; &sh[0]
%100 = OpSelect      %_ptr_Workgroup_float %93 %99 %98
%101 = OpLoad %float %100
```

Zero `OpBitcast` in the module, `spirv-val` passes, and the downstream
`OpAccessChain [0]` collapses away because the pointee is already the element
type.

#### 18. `Int64Atomics` declared with no feature enabled — FIXED

Found by the same run. The emitter declares the `Int64Atomics` SPIR-V capability
while `vk_device!` hardcoded `shader_buffer_int_64_atomics` and
`shader_shared_int_64_atomics` to `false`. A capability with no enabled feature
behind it is undefined (VUID-VkShaderModuleCreateInfo-pCode-08740). Now queried
from the device, which has to be right in both directions: enabling an
unsupported feature fails device creation, and leaving a supported one off is
what caused this. This device reports `true` for both.

#### 19. The regression test, and a vacuous first attempt

`046b1ed`. Verified in both directions: 3/3 with the fix, `Evaluated: 1 == 0`
with it disabled.

**My first attempt was vacuous and I nearly committed it** — the same failure
this session already found twice in DNNKernels (finding 14). A Tier 1
`compile_and_disasm` check on an equivalent hand-written function emits a
**valid** module and passed with the fix reverted. The illegal bitcast depends on
the exact KernelAbstractions `@kernel` lowering, not on the ternary. A pure
runtime assertion is no better, because the driver accepts the invalid module and
returns the right answer.

The committed test launches the real KA kernel, captures what the compiler
emitted via `LAVA_SPIRV_DUMP_DIR`, and walks the SPIR-V **binary** for an
`OpBitcast` (124) whose result type was declared by `OpTypePointer` (32) — binary
rather than disassembler text so it needs no `spirv-dis`. It also asserts the
dump is non-empty, since a dump never written would assert nothing.

### Suite state after the fixes

Full suite, `LAVA_VALIDATION=1`:

| | before | after |
|---|---|---|
| validation errors | 12 across 9 sites | **0** |
| test failures | — | 2 |
| unexpected passes | — | 6 |
| segfaults | 1 | 1 |

- **2 failures**, both `test_pipeline_cache_no_compile.jl:62,63` — finding 9, the
  negative control that does not fire on RADV.
- **6 unexpected passes**, all in the two miscompile families:
  `test_shared_index_division.jl:156` (x3) and
  `test_int32_cartesian_miscompile.jl:301,:334` (x3).
- **1 segfault**, now at `test_pipeline_cache_no_compile.jl:40` — a **third**
  distinct location for finding 16, which settles it as a floating GC race
  rather than anything test-specific. The suite still does not complete, so
  `test_static_workgroup.jl` did not run this time.

### All three "driver miscompile" suspects fail to reproduce here

This is the result to carry back to `lava-core` Phase 1, which lists two of them
as root-caused, mitigated and "cannot be settled on this hardware":

| item | test | on RDNA 3.5 |
|---|---|---|
| rank>=3 `Extruded` / static workgroup | `test_static_workgroup.jl` (13 asserts) | coverage 1.0 everywhere; the `min(1, b3/b2)` law does not hold |
| `OpUDiv` in a shared-store index | `test_shared_index_division.jl:156` | `@test_broken` **Unexpected Pass** x3 |
| narrow-index `CartesianIndices` | `test_int32_cartesian_miscompile.jl:301,:334` | `@test_broken` **Unexpected Pass** x3 |

Each of those tests was written to signal exactly this — *"Turns into a failure
the day it is fixed"*, *"if a driver update fixes it this turns into a failure,
which is the signal that `splitidx` could be relaxed"*.

Per Rule 0's own corollary this is evidence about **our** code, not exoneration
of the NVIDIA driver: our modules depend on something the spec leaves open which
one compiler exploits and the other does not. And that reading is now much
better supported than it was this morning, because findings 17 and 18 show the
same shared-memory paths were emitting **invalid** SPIR-V — an illegal pointer
bitcast and an unbacked capability — in modules nobody had ever validated.

Recommend re-opening all three against our own compiler, with `spirv-val` and
GPU-AV on, before anything is reported upstream.

## 2026-08-02, later — the vkCmdCopyBuffer fault is a GC lifetime bug

Rebased onto the merged `sd/nvidia` (`046b1ed`); the four fixes are upstream.

### A fast reproducer

The fault used to need a 15-minute suite and moved between tests. It reproduces
in **three files in one process, about two minutes**:

```
test_pipeline_cache_no_compile.jl
test_static_workgroup.jl
test_closesthit_via_rayquery.jl     <- SIGSEGV early in this one
```

Always at `Vulkan.cmd_copy_buffer`, `command.jl:1549`. The full path is now
known, and it is the **staging** path rather than anything kernel-related:

```
upload!          memory.jl:1644
copy_buffer!     memory.jl:1594 -> :1633     (host -> staging -> device)
cmd_copy_buffer! command.jl:1504 -> :1549
  -> vkCmdCopyBuffer -> SIGSEGV inside libvulkan_radeon.so
```

### It is a GC/finalizer bug — proved by elimination

Same reproducer, one line changed:

| | result |
|---|---|
| default | SIGSEGV at the third file |
| `GC.enable(false)` | **exit 0, all five files complete** |

That settles the class. It is not a driver defect, not recording order, and not
anything about the ray-query test it happens to land in. A finalizer destroys
something that a recorded-but-unsubmitted copy still refers to.

### Why Lava's own use-after-free scanner cannot see it

`FREED_BDA_SCAN_ENABLED` checks, before every destroy, whether the buffer's BDA
still appears in a live arg slab. Run over the crashing files it reports **zero
hits**, and that is structural rather than reassuring: a **staging buffer is a
transfer source, never a kernel argument**, so its address never enters an arg
slab at all. The scanner is blind to this entire class by construction.

Two warnings for anyone else reaching for it:

- It is far too slow for a full-suite run. `scan_arg_slabs_for_bda!` is linear in
  arg-slab size and runs on every destroy; with Tier 4 GPUArrays it took **50
  minutes to get from Tier 1 into `reductions`** and never reached the crash. Use
  it on a targeted file list.
- Both it and `FREE_DEBUG_LOG` only print at exit, which a SIGSEGV skips. Drain
  them from a `Timer` if you want the entries that precede the fault.

### The destroy trace, and the actionable detail

With `FREE_DEBUG_ENABLED` drained incrementally, the last twelve destroys before
the fault are all:

```
pool=true   lw=SET   active_recording=false
```

`active_recording = false` is the lead. `vk_free!` defers destruction when
`bq.active_batch !== nothing && bq.active_batch.recording`; here that check said
"no batch is recording", so the buffer was destroyed inline — while a copy that
refers to it is about to be, or has just been, recorded.

That is exactly the residual `vk_free!`'s own comment anticipates:

> *"Not certainly the last of it ... either a second window exists or something
> rarer shares this one. If it recurs, the next thing to check is whether a
> buffer can be reached by an open batch through something `pins` does not count
> either."*

A staging buffer reached through `get_staging`'s **raw tuple** is precisely such
a path. `get_staging` (`memory.jl:1526`) returns
`(buf.buffer, buf.memory, buf.mapped_ptr, buf.size)` — handles snapshotted off
the `VkManagedBuffer`, with no reference to the owner. Holding the tuple does not
keep the owner alive, and `destroy_buffer!` calls `buf.buffer.destructor()`
**explicitly**, so holding the Julia `Vulkan.Buffer` wrapper does not protect the
underlying `VkBuffer` either.

### Ruled out, so nobody re-walks it

**The pool recycle path is not the `mapped_ptr` divergence.** It looks like the
answer: `return_to_pool!` (`memory.jl:1497`) reuses the `VkManagedBuffer` object,
zeroing `mapped_ptr`, while `pool_alloc`'s free-list path restores `address`,
`size` and `state` but **not** `mapped_ptr`. That reads as a textbook divergence
and it is not one: pooled chunks are constructed with `mapped_ptr = 0`
(`memory.jl:1399`) and `PoolBlock` holds no mapping, so a pooled buffer is never
mapped. Zeroing an already-zero field is a no-op and not restoring it is correct.

So the finding-10 divergence, if it is separate from this, lives in the
non-pooled `vk_alloc` path (`memory.jl:592`), the only place `mapped_ptr` becomes
non-zero.

### Relation to the desktop's flush hang

Consistent with these being one bug, per the desktop's suggestion. The shape here
is "a buffer is destroyed while an unsubmitted batch still refers to it"; a hang
on `vkWaitSemaphores` is what the same defect looks like when the batch does get
submitted and waits on a timeline value for work referencing freed memory.

One structural gap supports that reading and **cannot be tested here**: `vk_free!`
consults `ctx.default_bq` only (`memory.jl:732`). A buffer reachable from a
recording batch on a different `BatchQueue` is invisible to the deferral check.
This device reports `max_queue_count = 1`; the desktop has `async_queue_count` 4
and the RT/async paths are where a second queue appears. If the desktop's
reproducer uses one, that is the first thing to instrument.

## 2026-08-02, Phase 3 — the first complete suite table on RDNA 3.5

Rebased onto `dev/Lava` `sd/nvidia` @ `83235f0` and `dev/JuliaVision` main @
`dd72061`.

### The suite completes, and the crash was `vk_reset_device!` — confirmed here

The desktop's diagnosis is right, and this machine reproduced the mechanism in
full before the fix arrived:

```
test_pipeline_cache_no_compile.jl (runtests.jl:171)  calls vk_reset_device!
  -> old context replaced, device_lost NEVER set
  -> pre-reset buffers still hold last_write = (old_bq, val)
test_static_workgroup.jl (runtests.jl:175)           GC finalizes one
  -> vk_free! (memory.jl:754) checks device_lost(bq.ctx) -> FALSE
  -> query_timeline -> vkGetSemaphoreCounterValue on a destroyed VkDevice
  -> SIGSEGV
```

`vk_reset_device!` contained no `mark_device_lost!` call at all, while its own
comment asserted the opposite: *"Pre-reset VkManagedBuffers hold a strong ref to
the OLD ctx whose `device_lost` is already true"*. That holds only when the reset
was **triggered by** a DEVICE_LOST. A proactive reset leaves the flag false.

This retro-explains everything this project could not close earlier: why the
crash moved between tests (it fires wherever GC runs after the reset), why the
3-file reproducer worked (the FIRST file was the one resetting), why
`GC.enable(false)` cured it, why the BDA scanner was clean (it hunts live
references, and nothing is wrong with the buffer's address), and why validation
reported no object-lifetime error (the `VkBuffer` is fine — the semaphore and
device are dead).

**Correction to this report's own finding 16.** The earlier "staging buffer raw
tuple in `get_staging`" hypothesis was WRONG. The `vkCmdCopyBuffer` stack was one
downstream symptom of the same dead device, not a lifetime bug in `get_staging`.

With `mark_device_lost!(old)` added to the retire block, the suite printed a
summary for the first time:

```
Lava.jl | 23783 passed | 15 failed | 13 errored | 23811 total | 27m02s
```

### The table

**15 failures — 13 known, 2 new**

| site | count | status |
|---|---|---|
| `test_static_workgroup.jl:133,143,144,155,287,288` | 13 | the `Extruded` characterization tests; finding 11, they do not reproduce here |
| `test_disk_cache.jl:121,127` | 2 | **new**, see below |

**13 errors — 6 known, 7 one cause**

| site | count | status |
|---|---|---|
| `test_shared_index_division.jl:196` | 3 | `@test_broken` Unexpected Pass; the OpUDiv item |
| `test_int32_cartesian_miscompile.jl:301,334` | 3 | `@test_broken` Unexpected Pass |
| `test_disk_cache.jl:128-132` | 6 | per-device cache refactor |
| `test_struct_broadcast.jl:80` | 1 | per-device cache refactor (desktop already has this) |
| `test_subgroup_size_pinning.jl:90` | 1 | per-device cache refactor — **fixed**, `e8fd416` |

**The last three rows are one bug in three files**, not three bugs: tests that
index the now-per-device caches directly get the inner dict back.

- `test_disk_cache.jl:128` — `linked.compiled` where `linked` is now
  `Dict{Any,LavaLinkedKernel}`; the two failures at `:121,:127` are almost
  certainly the same shape. **8 sites, left alone** in case the desktop's
  working tree already covers them alongside the `struct_broadcast` fix.
- Mine — `DEVICE_SUBGROUP_SIZE` became `Dict{UInt64,Int}` keyed by `ctx.id`
  rather than a `Ref`. Fixed by querying through `device_subgroup_size(ctx)`
  then overriding by id. Back to 10/10.

### Two devices work here, and that is what the last segfault is

`twodevice_probe.jl`: **PASS**, exit 0, no segfault.

```
gpu id=1  AMD Radeon 8060S Graphics (RADV STRIX_HALO)
cpu id=2  llvmpipe (LLVM 22.1.8, 256 bits)
  gpu: dispatch ok   reduce ok   gemm ok
  cpu: dispatch ok   reduce ok   gemm ok
PIPELINE_CACHE grew by 12   (1 would mean the devices shared a pipeline)
LINKED_KERNEL_CACHE device keys: [0x1, 0x2]
LAUNCH_PLAN_CACHE   device keys: [0x1, 0x2]
```

So GUARDRAILS §8 holds on the RADV + lavapipe pairing, and the per-device keying
is real rather than nominal.

**The one remaining segfault follows from that.** It is now AFTER the summary,
and it is not RADV:

```
pthread_mutex_lock -> libvulkan_lvp.so        <- lavapipe
  vkGetSemaphoreCounterValue                  <- the same call as before
```

The suite includes `twodevice_probe.jl`, so it creates context id=2. The retire
fix marks `VK_CONTEXT_REF[]` lost — which the lavapipe context never is. At exit,
finalizers for lavapipe-backed buffers reach `query_timeline` on a device being
torn down, and it is the identical bug one context over.

**Suggested shape of the fix:** retire every live context, not just the global
ref. Now that contexts are per device and enumerable by `ctx.id`, "the context
being replaced" is no longer the same thing as "every context whose buffers are
about to be finalized".

### Not runnable here yet

Three items in the Phase 3 prompt reference work that is not on the pushed
branches, verified rather than assumed:

- **`BRIEF.md` "Phase 3"** — the portability brief is still Phases 1-2
  (`3803955`). The only Phase 3 in `plans/` is `lava-core`'s, about missing
  SPIR-V instructions.
- **A coopmat pipeline refusing to build without a 32-lane pin** —
  `pipeline.jl:380` is still the permissive form, so it silently skips pinning.
  The underlying question is already answered here regardless: size control is
  min 32 / max 64 / compute true, and a pinned width is honoured at BOTH widths
  (`test_subgroup_size_pinning.jl`, 10/10), so the refusal path will not fire on
  this device.
- **`flashepad` / `flashrpad`, `sdpaflashcm!(; epad, rpad)`** — absent;
  `git log --all -S flashepad` finds nothing, and the current keywords are
  `ballast`, `shpad`, `nrsc`, `preonly`, `rscbar`.

## 2026-08-02, independent work — LDS banking, and the decode gap

### RDNA 3.5's LDS has 32 banks, and conflicts follow gcd(stride, 32)

Measured, because this is the substance under `epad`/`rpad`: those knobs pad a
shared tile's row stride to break bank conflicts, and the shipped values were
tuned on NVIDIA. The bank count and the conflict-free strides are a hardware
property and can be measured without the knobs existing.

Every lane reads `lds[(lid*STRIDE + r) & MASK]` in a long loop, 64-lane
workgroup, 32 KB of LDS. All lanes step together so the bank pattern is constant
and STRIDE alone decides which banks collide. Minimum of 11 timed reps, swept in
**both directions** so nothing carries an order bias, after warming until the
clock stopped climbing (600 -> 927 MHz; the first sweep ramped mid-run and its
absolute numbers were discarded — GUARDRAILS §6).

| stride | fwd ms | rev ms | x vs stride 1 | gcd(s,32) |
|---|---|---|---|---|
| 1 | 0.5416 | 0.5252 | 1.00 | 1 |
| 3 | 0.5441 | 0.5383 | 1.02 | 1 |
| 17 | 0.5536 | 0.5376 | 1.02 | 1 |
| 33 | 0.5486 | 0.5413 | 1.03 | 1 |
| 65 | 0.5532 | 0.5362 | 1.02 | 1 |
| 2 | 0.5506 | 0.5566 | 1.05 | 2 |
| 4 | 0.5889 | 0.5834 | 1.11 | 4 |
| 12 | 0.5768 | 0.5809 | 1.10 | 4 |
| 8 | 0.6496 | 0.6346 | 1.21 | 8 |
| 24 | 0.6336 | 0.6312 | 1.20 | 8 |
| 16 | 0.8446 | 0.7440 | 1.42 | 16 |
| 48 | 0.7556 | 0.7492 | 1.43 | 16 |
| 32 | 0.9812 | 0.9651 | 1.84 | 32 |
| 64 | 0.9820 | 0.9700 | 1.85 | 32 |

**The cost is a function of `gcd(stride, 32)` and nothing else** — 48 matches 16,
24 matches 8, 12 and 20 match 4, and every odd stride is free. That is 32 banks.

Consequence for the padding knobs, whatever they end up called: **any odd row
stride is conflict-free on this device.** Padding a power-of-two row stride by
one element is sufficient and there is nothing to gain from padding further. A
stride that stays a multiple of 32 is the worst case and costs ~1.85x on this
probe.

The absolute ratios understate a true k-way conflict (stride 32 puts all 64 lanes
on one bank and costs 1.85x, not 32x) because the loop also pays its own
arithmetic and wave64 issues LDS in halves. The **ordering** is what is being
claimed here, and it is unambiguous and reproducible in both sweep directions.

### SAM 2: the decode gap is nearly 3x worse than the encode gap

The earlier 294 ms figure did not move after flash-decoding landed, and the
reason is a flaw in **my** measurement, not in K2: `runsam2` is encode + decode,
encode dominates, so a 3.55x decode win is invisible in the total. Measured
separately:

| | Radeon 8060S | desktop (RTX 4000 Ada) | ratio |
|---|---|---|---|
| encode | **278.2 ms** | 100.4 ms | 2.8x |
| decode (the click path) | **17.28 ms** | 2.21 ms with K2 | **7.8x** |
| total | 295.5 ms | ~103 ms | 2.9x |

Encode is 2.8x off the desktop, which is roughly what a 40-CU iGPU on shared
memory should be. **Decode is 7.8x off**, nearly three times the encode gap.

That is the interesting number, and it points where the plan already suspects:
decode is where the coopmat2-gated kernels live, and `coopmat2` is **eight
`false`s** on this device, so the per-element flash path falls back wholesale.
Flash-decoding's benefit is structurally smaller here for that reason.

It makes an AMD fallback story for K3 (coopmat2 reductions) and K5 (tensor
addressing / block loads) worth more than the raw NVIDIA speedups suggest: on
this device those kernels are not choosing between fast and faster, they are
choosing between a fallback and nothing.

Caveats: synthetic input (a centred disc), one click, one click position. Frozen
cache 106 hits / 0 misses on every run. Model build 42.8 s.

### What that means for the flash-cm kernel, before the padding knob exists

`attn_flash_cm!` stages its tiles as (`flash.jl:409`)

```julia
qs = @localmem Float16 (EP * BR,)      # (e, r) at r*EP + e
```

so the shared row stride is `EP` elements, and

```julia
EP = cld(E, dev.tile) * dev.tile       # flash.jl:1113, :1240
```

is the head dim rounded up to the cooperative-matrix tile, with `flashcmfits`
requiring `EP % dev.tile == 0`.

`qs` is `Float16` and an LDS bank word is 4 bytes, so a row stride of `EP`
elements is **`EP/2` bank words**. `EP` is always a multiple of 16, therefore
`EP/2` is always a multiple of 8, therefore `gcd(EP/2, 32) >= 8` **always**.
Combining that with the measured table above:

| head dim E | EP | bank words | gcd(.,32) | measured cost |
|---|---|---|---|---|
| 64 | 64 | 32 | **32** | **1.84x** |
| 72 | 80 | 40 | 8 | 1.21x |
| 96 | 96 | 48 | 16 | 1.42x |
| 128 | 128 | 64 | **32** | **1.85x** |

**Head dims 64 and 128 — the two most common — land on the worst possible stride
on this device**, every row of every staged tile starting on the same bank.

Two consequences:

1. It explains the desktop's `epad` result from the hardware up. "epad is worth
   -31.4% at head dim 64" is what breaking a `gcd = 32` collision looks like, and
   the same collision exists here, so the knob should pay at least as well.
2. **The tile constraint means the kernel cannot reach conflict-free by tuning
   `EP`.** Any legal `EP` is a multiple of 16 and so is stuck at `gcd >= 8`. The
   fix has to decouple the shared **storage** stride from the coopmat **tile**
   extent — store rows at `EP + pad` while still loading 16-wide tiles — which is
   presumably exactly what `epad` does. Padding by one Float16 element makes the
   stride odd in bank words and is measured free.

Not claimed: an end-to-end number. `EP` is derived and there is no knob on this
branch to vary it, so this is the hardware measurement plus the kernel's own
arithmetic, not an A/B of the real attention. When the knob lands this is a
confirmation run rather than a search: predicted best is any `EP` with `EP/2`
odd, and predicted worst is the current default at head dims 64 and 128.

`shpad`, which the branch does have, is **not** this knob — it is an occupancy
ballast (`flash.jl:425`, a dummy `@localmem` touched at `:898` so it survives
optimisation), and sweeping it answers a different question.

### Finding 1 lands in practice: `dev.subgroup` vs `dev.coopmatsubgroup`

The six asset-independent DNNKernels tests, re-run against the post-refactor
`ctx`/plan API (nothing had run them here since `Ctx` and the plan objects
arrived). Five green. `test_flash.jl` failed **4 assertions**, all one cause, and
all of them pass on wave32 hardware because the two fields coincide there:

```
:167  (BR * 72) % (NW * dev.subgroup) == 0    ->  256 == 0, 128 == 0   (x3)
:348  p.NT == p.NW * dev.subgroup             ->  128 == 256
```

The device **default** here is 64; Lava pins a cooperative-matrix module to 32.
So `dev.subgroup` overstates the workgroup by 2x, and every shipped tiling
"failed" its divisibility check.

**The root is a `src/` comment**, which is why this was not only a test fix.
`kernelplans.jl:58` documented the field as

```julia
NT::Int          # NW * dev.subgroup — never `NW * 32`
```

The code has always used `dev.coopmatsubgroup` (`flash.jl:1116`, `:1250`), and
`flashcm_tiling`'s docstring already says which of the two is wanted: *"the width
a cooperative-matrix module actually runs at, which Lava pins to 32. The tiling
needs the second."* The `never NW * 32` clause was a correction to the old
hardcoded literal that overcorrected onto the wrong field, and **both assertions
were written from it**. Fixed, and it now says why.

216/216 fused attention, 15/15 plans. Note the second failure was invisible until
the first was fixed — the file threw before reaching it.

This is portability finding 1 arriving in real code: `device_subgroup_size()` is
a **default**, not a fact about the kernel that is about to run, and anything
conflating the two is correct only on hardware where they coincide. The desktop
added `dev.coopmatsubgroup` precisely because of that finding; the comment and
the tests were what did not catch up.

**Swept for the same class elsewhere — clean.** `Lava/src/array/gemm.jl:270` is
*correct* and worth citing as the pattern to copy:

```julia
device_subgroup_size(ctx) == GEMM_SUBGROUP || can_require_subgroup_size(ctx, GEMM_SUBGROUP)
```

with a comment that names both ways to have a 32-lane subgroup. The remaining
literals in `attn_flash_cm!` (`NW * 32` at `flash.jl:408`, `:440`, `tid ÷ 32` at
`:455`) are correct **by the pin**, not by accident of this device — which is
exactly why the "refuse to build a coopmat pipeline where 32 lanes cannot be
pinned" change matters: those literals are silently wrong on any device that
cannot pin, and nothing would report it.

### Post-refactor status of the standalone DNNKernels tests

| file | result |
|---|---|
| `test_constfold.jl` | 19 / 19 |
| `test_foldoutcasts.jl` | 2 skips recorded (no `gen/` tree; see finding 14) |
| `test_convtranspose_gemm.jl` | 78 / 78, including the new 1x1-as-GEMM testset |
| `test_transposeLE.jl` | 16 / 16 |
| `test_coopmat_attention.jl` | 11 / 11 |
| `test_flash.jl` | **216 + 15 after the fix**, was 4 failures |

### The lavapipe teardown fault: four attempts, not reproduced

Recorded so nobody repeats the same three ideas. All exit 0:

1. Two contexts, work on lavapipe, `vk_reset_device!`, forced GC.
2. Same, but forcing the finalizer **ordering** — collect the context first so the
   lavapipe `VkDevice` is finalized before the buffers that reference it.
3. The suite's real sequence: the probe creates ctx 2 (`runtests.jl:190`) and a
   later test resets (`test_gpuav_clean.jl`, `:489`), with buffers left **live**
   so the finalizers run at process exit, which is where the suite faults
   (`in expression starting at none:0`).

So "the lavapipe context is never retired" is **not sufficient** as an
explanation. The suite still reproduces it reliably, so the repro exists; it
needs more of the suite's accumulated state than these isolate.

### GPUFiltering: 65 / 65 on AMD, first coverage of that package here

Never run on this machine before. All green, and not merely smoke tests —
`gaussianblur!` is validated against `ImageFiltering`'s CPU reference, so this is
a numerical check.

| testset | |
|---|---|
| `coloradjust!` | 3 / 3 |
| `gaussianblur!` vs ImageFiltering | 2 / 2 |
| `unsharpmask!` | 2 / 2 |
| `opticalflow!` | 4 / 4 |
| `fitaffine` | 14 / 14 |
| `fithomography` | 6 / 6 |
| `warp!` | 4 / 4 |
| `samplewindow!` / `nccpeak` | 7 / 7 |
| `fitsimilarity` / `PatchTracker` | 23 / 23 |

Run it with `Pkg.test("GPUFiltering")`, not
`julia --project=<workspace> GPUFiltering/test/runtests.jl`. The test script does
`using FixedPointNumbers`, which is a dependency of the *package* and therefore
only an indirect dependency of the workspace project, and a script can only
`using` direct ones. The error it produces —
`Package FixedPointNumbers not found in current path` — reads like a missing
declaration and is not one.

**Correction to something recorded earlier in this session.** I diagnosed the
same-looking `SAM2Runner` failure as a stale manifest entry, and for that package
it genuinely was: its `[[deps.SAM2Runner]]` block had no `deps` line at all and
`Pkg.resolve()` treated that as consistent, so `Pkg.rm` + `Pkg.develop` was the
fix. I then assumed `GPUFiltering` was the same and it was **not** — its manifest
entry lists `deps` correctly and `FixedPointNumbers` is present. Two different
causes behind one error message; check the manifest entry before assuming which.

**Confirmed as a set.** All six re-run together after the `coopmatsubgroup` fix:
exit 0, **367 assertions, zero failures**.

```
constfold 19 · foldoutcasts 2 (skips, no gen/ tree) · convtranspose_gemm 78
transposeLE 16 · coopmat_attention 11 · flash 216 + 11 + 15
```

That is the whole asset-independent DNNKernels surface green on RDNA 3.5
post-refactor. It was not, before the `dev.subgroup` correction.

**Plus a seventh I had missed.** My standalone list predated the refactor, which
added `test_diagnostics.jl` — host-only on the CPU backend, so it runs anywhere,
and it asserts the property that replaced the five module-level `Ref`s
(`OPTIMES`, `OPDOUBLE`, `OPDOUBLEFILTER`, `PLAN_MISSES`, `LAUNCH_PROBE`): two
runs in one process can be instrumented differently and neither sees the other's
measurements. **13 / 13**, bringing the standalone total to **380**.

## 2026-08-03 — GPU-assisted validation, the other half of Rule 0 instrument 1

`spirv-val` (via `LAVA_VALIDATION=1`) found two bugs earlier. **GPU-AV had never
been turned on**, and Rule 0 names both: *"`spirv-val --target-env vulkan1.3`,
then GPU-assisted validation (`Lava.enable_gpu_av`). Cheap, and neither is run by
default."*

### It found a third bug immediately

```
vkCreateShaderModule(): OpGroupNonUniformShuffle is using a 64-bit int scalar but
VkPhysicalDeviceShaderSubgroupExtendedTypesFeatures::shaderSubgroupExtendedTypes
was not enabled.                          VUID-RuntimeSpirv-None-06275
```

SPIR-V group operations on 8-, 16- or 64-bit types require that feature. Lava
binds `subgroup_shuffle` for `Int64`/`UInt64`/`Float64` and `vk_device!`
hardcoded it `false`, so those shuffles were undefined — while
`test_subgroup_shuffle.jl` passed **520/520**, because the driver ran them anyway.

Fixed by querying the device (`559d499`). Now 520/520 *and* defined.

**Three for three, one family:** a capability the emitter uses with no enabled
feature behind it, invisible with validation off, correct answers on the driver.
`be52353` (invalid logical-pointer `OpBitcast`), `332dc33` (`Int64Atomics`),
`559d499` (`shaderSubgroupExtendedTypes`).

### First fully crash-free full-suite run on this machine

```
23793 passed | 15 failed | 11 errored | 0 broken
segfaults: 0                validation errors: 10
```

15 failures and 11 errors, all previously accounted for: the `Extruded`
characterization tests and the `@test_broken` unexpected passes. The three
per-device-cache test breakages are fixed and gone.

### Four distinct VUIDs, of which two are real

| VUID | count | verdict |
|---|---|---|
| `VkShaderModuleCreateInfo-pCode-08737` | 4 | **real — my earlier fix was incomplete** |
| `RuntimeSpirv-PhysicalStorageBuffer64-11819` | 2 | **needs corroboration**, see below |
| `VkBufferCreateInfo-size-06409` | 2 | benign |
| `vkAllocateMemory-pAllocateInfo-01713` | 2 | benign |

**Benign, both the same event:** a deliberate 40 GB buffer/allocation against a
34 GB heap. That is an OOM-handling test doing its job.

#### My `OpBitcast` fix was incomplete

```
%371 = OpBitcast %_ptr_Function_uint %160
```

**Function** storage class this time, not Workgroup, reached from GPUArrays'
`sliced setindex, CPU source` (`indexing.jl:77`).

`be52353` fixed the **`OpSelect` reconciliation** path only. I had already
identified three pointer-`OpBitcast` sites — `emit.jl:1362`, `:1378`, `:1414` —
and fixed one. The other two are in the **load** path, where the emitter bitcasts
a pointer to change its pointee type for a differently-sized load, and they are
still live. This is proof they are reachable from a real test rather than
theoretical.

Note the load path may not be fixable by the same "drill with `OpAccessChain`"
trick: drilling reaches an aggregate's element, but a narrowing/widening load
wants a *differently typed* pointer to the same address, which the Logical
addressing model does not provide. That probably needs the load itself to change
(load the pointee's true type, then convert the value), which is what the
existing widening branch at `:1335` already does for one case.

#### A GPU-AV out-of-bounds write, on lavapipe

```
vkCmdDispatch(): Out of bounds access: 1 bytes written at buffer device address
0x7f1528703a7f. Stage = Compute. Global invocation ID (x, y, z) = (0, 0, 0)
```

**Not claimed as an emitter bug yet.** It appears immediately after a lavapipe
context is initialised, i.e. inside `twodevice_probe.jl`, so it is on the CPU
device rather than RADV. GPU-AV also reports that it forced several features on
for its own instrumentation (`fragmentStoresAndAtomics`,
`vulkanMemoryModelDeviceScope`, `storageBuffer8BitAccess`, a
`VkPhysicalDevice16BitStorageFeatures` added to the chain), so the device
configuration under GPU-AV is not the configuration under test.

A one-byte overrun at global invocation `(0,0,0)` is a specific enough signature
to chase, and it is exactly what GPU-AV exists to catch. But it needs
reproducing on RADV, or on lavapipe without GPU-AV's forced features, before it
is a finding rather than an artefact.

### Operational note

GPU-AV prints its own warning and it is right: *"Both GPU Assisted Validation and
Normal Core Check Validation are enabled, this is not recommended as it will be
very slow."* Run core validation first, fix what it reports, then GPU-AV alone.

### The second `OpBitcast` family, reproduced and inventoried

`GPUFiltering`-free reproducer, 30 seconds, no GPU-AV needed — just dump and
validate:

```julia
for T in (Int16, Int32, Int64, Float16, Float32, Float64,
          ComplexF16, ComplexF32, ComplexF64,
          Complex{Int16}, Complex{Int32}, Complex{Int64})
    x = Lava.LavaArray(zeros(T, (2,3,4)));  y = Lava.LavaArray(rand(T, (2,3)))
    x[:, :, 2] = y
    @assert Array(x[:, :, 2]) == Array(y)
end
```

with `LAVA_SPIRV_DUMP_DIR` set, then `spirv-val` each dump. **All twelve element
types produce correct results and two `gpu_getindex_kernel` modules are invalid.**

Real element types alone do **not** reproduce it — the first eight I tried were
all clean. It needs the `Complex` types, which is why it surfaced through
GPUArrays' `supported_eltypes` and not through anything hand-written.

The offending pattern is a **narrowing store into a Function-scope alloca**:

```
%160 = OpVariable %_ptr_Function__arr_ulong_uint_2 Function   ; [2 x ulong]
%373 = OpBitcast %_ptr_Function_uint %160                     ; ptr[2 x ulong] -> ptr uint
       OpStore %373 %336
%376 = OpAccessChain %_ptr_Function_ulong %160 %uint_0        ; legal drill to element 0
%379 = OpBitcast %_ptr_Function_uint %376                     ; then illegally re-type ulong* -> uint*
       OpStore %379 %339
```

Note the second one: the emitter *does* drill legally with `OpAccessChain`, then
bitcasts the result anyway because the element is `ulong` and the value is
`uint`. Drilling cannot fix this one — the address is already right and it is the
**width** that is wrong.

#### Corrected inventory of pointer `OpBitcast` sites

An earlier entry in this report said "the emitter has three pointer `OpBitcast`
sites (`emit.jl:1362`, `:1378`, `:1414`)". That was wrong twice: `:1414`/`:1416`
bitcasts a loaded **value**, which is legal, and the two **store-path** sites were
missed entirely.

| site | operand | status |
|---|---|---|
| `~emit.jl:6067` OpSelect reconciliation | pointer | **fixed**, `be52353` |
| `emit.jl:1362`, `:1378` load path | pointer | **live** |
| `emit.jl:2174`, `:2186` store path | pointer | **live** — what this reproducer hits |
| `emit.jl:1416`, `:2089` | value | legal, not a defect |

#### Why the remaining four need a different fix from the one that worked

`be52353` worked because the `OpSelect` case was an **addressing** mismatch:
array-of-T versus T at the same address, and `OpAccessChain` expresses exactly
that. The load and store paths are **width** mismatches — a 32-bit value through
a pointer whose pointee is 64-bit — and the Logical addressing model has no way
to produce a differently-typed pointer to the same address at all.

So the fix has to move to the value side: load the pointee's true type and
convert, or widen the value and store the full pointee. `emit.jl:1335` already
does exactly this for the widening-load case, with a comment explaining that
bitcasting the pointer there would read past the field. That branch is the
template; the narrowing and store cases need the same treatment.

Not attempted here: a narrowing store that only wants the low 32 bits of a
64-bit slot is not semantically identical to a widening store of the whole slot,
so this is a change to what the kernel writes and needs the emitter's owner, not
a portability pass. The reproducer above is the thing to iterate against.

## 2026-08-03, Phase 3 continued — the `OpBitcast` family is closed

Pinned at Lava `e6456c0` (rebased onto `sd/nvidia` `0903c6f`), JuliaVision
`07e33e7` (merged with `sd/kernels-refactor` `e89e694`).

### Suite: 23793 pass / 13 fail / 9 error, zero segfaults, zero `pCode-08737`

Full run, `LAVA_VALIDATION=1`, 26m04s, 2710 modules compiled and dumped.

| | baseline (Phase 3.1) | now | change |
|---|---|---|---|
| failures | 15 | **13** | `test_disk_cache.jl:121,127` fixed |
| errors | 13 | **9** | `test_disk_cache.jl:128-132` + `test_subgroup_size_pinning.jl:90` fixed |
| `pCode-08737` | 4 | **0** | the store-path fix |
| segfaults | 0 | 0 | |

The 13 remaining failures are all `test_static_workgroup.jl` (`:133,:143,:144,`
`:155,:287,:288`) — the `Extruded` characterization tests, which assert that a
driver bug *still reproduces* and therefore fail on a driver that does not have
it. The 9 errors are `test_shared_index_division.jl:196` (3),
`test_int32_cartesian_miscompile.jl:301,334` (3) — all `@test_broken` Unexpected
Pass, same cause — plus `test_struct_broadcast.jl:80`, which `bca7906` fixes and
which the rebase has now brought in.

### Both remaining `OpBitcast` sites fixed, and one of them is unexercised

`67b55ac`/`effd47e` (store) and `60364e4` (load) close the inventory. Both
reconcile on the **value** side, which is the part the earlier entry in this
report got right in theory: load the slot at its true width, `OpUConvert` (SPIR-V
defines it as truncate-or-zero-extend, so one instruction serves both
directions), `OpBitcast` the *value* if a float was wanted.

The store side's narrowing case is a read-modify-write — extend, load slot, mask,
OR, store — so the high bits of the slot survive exactly as the old narrow
pointer store left them. That was the objection the earlier entry raised against
attempting it, and it is answered rather than waived: the write is
semantics-preserving, not merely legal.

**The load-path sites are not reachable by anything available here, and this is
a negative result, not a verification.** A probe on both branches recorded:

| driver | modules | hits |
|---|---|---|
| full Lava suite | 2710 | **0** |
| mirror of the store reproducer (`z = x[:, :, 2]`, twelve element types) | 24 | **0** |

So `emit.jl`'s `pointee_ty isa IntegerType` load branch — including the
pre-existing *widening* case, which carries a comment about RADV rejecting the
alternative — is dead in this suite. The fix is correct by construction and by
symmetry with the store side; it is not correct by measurement, and a green suite
must not be read as having exercised it. The store side, by contrast, went from
2 invalid modules to 0 on a reproducer that runs in 30 seconds.

Two things fell out of unifying the branches, both worth having independently:
`i1` maps to `OpTypeBool` and so is reachable by neither convert nor bitcast
(`trunc iN to i1` is emitted as `(x & 1) != 0`); and the widening branch passed a
bare `Aligned` memory operand where every other Workgroup load in the file passes
`Aligned | NonPrivatePointer`, which is meaningful because Lava emits the Vulkan
memory model.

### A handled allocation failure was blaming innocent code — `dfb797e`

**New bug, and it is a diagnostic-quality bug, which is the kind that costs the
most time.** The suite reported a failure in
`test_source_mapping.jl:739`, "Source mapping doesn't break kernel execution",
as a `LavaError during vk_flush!` on a **four-element** `Float32` upload — with a
message about a **40 GB** buffer.

That test is innocent. `test_source_mapping.jl:699`, forty lines earlier,
deliberately asks for 40 GB against a 34 GB heap:

```julia
Lava.try_vk_alloc(Lava.vk_context().default_bq, 40_000_000_000)  # 40GB, will fail
```

`try_vk_alloc` handles the refusal and returns `AllocFailure`, and it *tried* to
absorb the validation messages — but it called `empty!(VALIDATION_MESSAGES)`
without calling `drain_validation_messages!()` first. The callback writes into a
lock-free ring (`VAL_RING_*`), and only the drain moves entries out of it, so the
messages stayed in the ring until the next `check_validation_errors!` — which
belonged to unrelated code, and threw there.

One line. The correct idiom was already in the file: `clear_printf_output!` does
`drain_validation_messages!(); empty!(PRINTF_MESSAGES)`.

**This corrects an earlier entry in this report.** The section "Four distinct
VUIDs, of which two are real" called `VkBufferCreateInfo-size-06409` and
`vkAllocateMemory-pAllocateInfo-01713` *"benign — an OOM-handling test doing its
job"*. Benign as Vulkan **events**; not benign as test **outcomes**, because they
abort a later unrelated testset whenever validation is on. Only two of the four
VUIDs needed fixing, but all four needed *handling*.

`test_tolerated_alloc_failure.jl` asserts the **ring** is empty after a drain,
not just the list — draining is precisely what the buggy version skipped — and
then reproduces the original symptom with the small unrelated upload. Verified
to fail with the fix reverted: 1 passed, 1 failed, 1 errored, the error being the
same `LavaError during vk_flush!`.

### Phase 3.3 — the coopmat 32-lane pin: the assumption holds

```
device            : AMD Radeon 8060S Graphics (RADV STRIX_HALO)
default subgroup  : 64
size control      : min=32 max=64 compute=true
can pin 32        : true
```

**The guard does not fire here**, which is what the brief predicted and asked to
have confirmed. `test_coopmat_subgroup_refusal.jl`, 6/6.

The positive control is the part worth reporting, because the first version of it
**passed vacuously**. Driving the refusal through `mul!` with the capability
cache faked to a wave64-only device produced no error at all: `mul!` returned
normally and computed the right answer. `coopmat_gemm_available` consults
`can_require_subgroup_size` itself (`gemm.jl:271`), so the faked cache makes it
answer false and the call routes to the scalar GEMM without ever asking for a
cooperative-matrix pipeline.

**So the refusal is a backstop behind that gate, not the first line of defence,
and it only protects callers that do not ask `coopmat_gemm_available` first.**
The test now launches a kernel that touches a cooperative matrix directly, which
is what makes the module declare `CooperativeMatrixKHR` — the condition
`get_compute_pipeline` actually keys on — and the refusal fires with its own
message.

### DNNKernels op coverage runs here; the reference comparison still cannot

The blocker was recorded as "no `matanyone-refs` artifact exists". More precisely:
the `matanyone` artifact **is** installed
(`~/.julia/artifacts/9369a615…`) and carries `graphs/` and
`weights.safetensors`, but **not** `refs-*.safetensors` or
`refs_manifest-*.json`, and its graphs are the pre-split flat layout
(`graphs/encode_image.json`) rather than the `graphs/aten-$precision/` the test
wants. Regenerating the refs needs `tools/dump_refs.py`, i.e. PyTorch and the
model, which the brief rules out.

The half that does not need refs runs, and is worth having:

| graph | ops | missing |
|---|---|---|
| `encode_image` | 8 | none |
| `transform_key` | 5 | none |
| `encode_mask_deep` | 24 | none |
| `encode_mask_shallow` | 23 | none |
| `pixel_fusion` | 9 | none |
| `pred_uncertainty` | 7 | none |
| `segment` | 13 | none |
| `readout_query` | 38 | none |

**Full op coverage on all eight graphs, nothing missing.** That is the first
coverage check of MatAnyone's graphs on this machine, and it says the gap is
purely the reference activations, not the kernel library.

### One more hardcoded 32, in a test

`DNNKernels/test/test_flash.jl:129` gated on `NW * 32 <= Lava.WORKGROUP_LIMIT[]`.
The literal was numerically right — Lava pins coopmat modules to 32, so that is
the real thread count — but it stops being a coincidence only once it names which
of `dev.subgroup` (64 here) and `dev.coopmatsubgroup` (32) it means. Now
`NW * dev.coopmatsubgroup`, the same distinction as finding 1.

Not fixed, filed: `flash.jl:472` computes `NT = NW * 32` **inside the kernel**,
duplicating `plan.NT`. Correct by construction today, since the pipeline now
refuses to build unless 32 lanes can be pinned, but it is a second source of
truth for a number the plan already carries.

## 2026-08-03, Phase 3.4 — the bank-conflict sweep, and a prediction of mine that was wrong

`epad × rpad` at `E ∈ (16, 32, 48, 64, 72)`, tiling 64x32/8, `Lq = Lk = 4096`,
8 head-batches — the same shape the NVIDIA numbers in `flashepad`'s docstring are
quoted on. 100 configurations, **all 100 numerically correct** (each is checked
against a CPU reference and for non-trivial output before it is allowed to
contribute a time; a kernel that writes nothing is fast and wrong).
Script: `bankconflict_sweep.jl`.

### The prediction, stated before the sweep ran, and falsified by it

The earlier LDS section of this report measured cost as a function of
`gcd(stride, 32)` and nothing else, then predicted:

> predicted best is any `EP` with `EP/2` odd, and predicted worst is the current
> default at head dims 64 and 128

`epad = 2` is exactly that stride: `(64+2)/2 = 33` bank words, `gcd(33,32) = 1`,
conflict-free by the arithmetic. **It is the slowest or second-slowest setting at
every head dimension tested.**

| E | epad=0 | **epad=2** (gcd 1) | epad=4 (gcd 2) | epad=8 (gcd 4) | epad=16 |
|---|---|---|---|---|---|
| 16 | 3.020 | **3.975** (4/5) | **2.893** (1/5) | 3.572 | 4.012 |
| 32 | 5.838 | **7.134** (5/5) | **4.799** (1/5) | 5.018 | 5.291 |
| 48 | 9.433 | **12.288** (5/5) | **8.831** (1/5) | 9.047 | 12.264 |
| 64 | 13.371 | **12.963** (4/5) | 8.856 | **8.786** (1/5) | 9.420 |
| 72 | 14.807 | **19.254** (5/5) | **13.867** (1/5) | 14.468 | 16.626 |

(ms, `rpad = 0`; rank within the head dim in brackets.)

The winner is always `gcd = 2` or `gcd = 4`, never `gcd = 1`.

**Why the earlier measurement did not transfer.** That stride sweep was a scalar
loop in which alignment was constant and only the bank pattern varied, so it
measured bank conflicts correctly *in isolation*. This kernel's LDS traffic is
wide vectorised cooperative-matrix loads, and there the dominant term is **row
alignment**, not bank spread:

| epad | stride (Float16 elements) | byte alignment of each row | result |
|---|---|---|---|
| 2 | EP+2 | 4 B | narrow/unaligned access — **worst** |
| 4 | EP+4 | 8 B | `ds_read_b64` — **best** |
| 8 | EP+8 | 16 B | `ds_read_b128` — close second, more LDS |
| 16 | EP+16 | 32 B | same width, more LDS still — loses on occupancy |

So the finding is not "the bank model is wrong here", it is that **I
over-extrapolated a scalar-stride measurement to a kernel that reads LDS in
vectors**. `epad = 2` buys a conflict-free bank pattern by making every row start
off a natural boundary, and pays more for that than the conflict ever cost. The
earlier measurement stands; the inference drawn from it did not.

### The shipped defaults are wrong here, in both branches of their own rule

`flashepad(EP) = EP % 32 == 0 ? 8 : 0` — tuned on NVIDIA — is beaten at every
head dimension:

| E | EP | shipped | best | delta |
|---|---|---|---|---|
| 16 | 16 | (0,0) 3.020 | (4,2) 2.038 | **-32.5%** |
| 32 | 32 | (8,0) 5.018 | (4,2) 3.479 | **-30.7%** |
| 48 | 48 | (0,0) 9.433 | (4,2) 7.172 | **-24.0%** |
| 64 | 64 | (8,0) 8.786 | (8,2) 6.790 | **-22.7%** |
| 72 | 80 | (0,0) 14.807 | (4,2) 11.372 | **-23.2%** |

Both branches of the rule are wrong on this hardware: where it pads by 8 the
better answer is 4, and where it pads by 0 the better answer is also 4. `epad=4`
wins outright at four of five head dims and loses to `epad=8` by 0.6% at E=64,
which is inside the noise.

**`rpad = 2` is the larger effect here, and it is the opposite of the desktop's
conclusion.** On NVIDIA `rpad` was worth -13.7% in this benchmark and ships OFF
because it cost the real encode +6.7%. Here it is worth **-18% to -30%**,
consistently, at every head dimension and on top of every `epad`.

That is precisely the result the brief warned not to trust from a sweep, so it is
not adopted on this evidence. The real-model check follows.

### The real-model check: the sweep's 22–32% does NOT transfer

`DNNKernels.encode` alone, interleaved A/B/A/B in one process, warm-up gated on
three consecutive runs agreeing to 2%, and a checksum over the encoder output so
a setting that is fast because it computed something else cannot pass as a win.

```
round 1  shipped  (flashepad rule, rpad=0)       min   279.98 ms   med   284.68 ms
round 1  sweep winner (epad=4, rpad=2)           min   275.51 ms   med   288.15 ms
round 2  shipped  (flashepad rule, rpad=0)       min   283.08 ms   med   290.97 ms
round 2  sweep winner (epad=4, rpad=2)           min   282.51 ms   med   290.06 ms

shipped       runs 279.98 / 283.08   spread 1.1%
sweep winner  runs 275.51 / 282.51   spread 2.5%
encode: 279.98 -> 275.51 ms   -1.6%      checksums agree: true
```

**-1.6% against a within-setting spread of 1.1% and 2.5%, and the medians cross.
That is no effect.** A 22–32% microbenchmark win produced nothing measurable on
the model, which is the same shape as the desktop's `rpad` result — there the
sweep said -13.7% and the encode said +6.7% — arriving at the same place from the
opposite direction.

So the defaults are **not** changed on this evidence, and the sections below show
that holds under per-kernel timing too, not just at the whole-encode level. The
brief's instruction to check the winner against a real model rather than the
sweep is the whole reason this section does not end with a patch.

### The knob works, the path is taken, and neither shows up in the model

Two controls, because "no effect" and "measured nothing" look identical:

1. **The path is taken.** 331 `flashepad` queries during one encode, every one at
   `EP = 80` — SAM 2's own head dimension, and the one the sweep covers.
2. **The knob reaches the kernel.** `epad = 2`, which the sweep calls catastrophic
   (+39% at E=72), moves the encode by +3.1%. It moves; it moves by a twelfth of
   what the sweep predicts.

(The first attempt at control 1 died with a `StackOverflowError` at 10463 frames:
`const orig = DNNKernels.sdpaflashcm!` names the **generic function**, not the
method, so the wrapper called itself. Instrumenting `flashepad` cannot recurse.)

Per-dispatch Vulkan timestamps then isolate the kernel from the ~300 other
dispatches it shares the encode with. **The flash kernel is 36% of the encode**
(103 ms of 283), so a real 20% win on it would be a visible 7% on the total.

| setting | flash kernel GPU ms (median of 5) | within-setting spread | vs shipped |
|---|---|---|---|
| shipped (`epad` rule, `rpad=0`) | 103.62 | 9.4% | — |
| `epad=2, rpad=0` | 104.96 | 8.3% | +1.3% |
| `epad=4, rpad=0` | 103.12 | 11.4% | -0.5% |
| `epad=8, rpad=0` | 105.26 | 7.4% | +1.6% |
| `epad=4, rpad=2` | 105.36 | 2.2% | +1.7% |
| `epad=2, rpad=2` | 106.32 | 7.8% | +2.6% |

**Every setting is within 2.6% of every other, against a within-setting spread of
up to 11.4%. There is no effect to measure.** A 22–32% difference in the isolated
kernel is worth nothing inside the model, and that is the result.

#### A near-miss worth recording, because I almost reported the opposite

The **first** profile ran one sample per setting and said the flash kernel went
99.43 → 80.59 ms at `epad=2` (**-19.0%**) and → 124.36 ms at `epad=8` (**+25.1%**)
— an apparent inversion of the sweep, and an apparent vindication of the
bank-conflict prediction the sweep had just falsified. It was noise. With n=1
against a spread that the repeat later measured at 7–11%, those numbers are
exactly what sampling that distribution once produces.

It was caught only because it contradicted the wall-clock A/B, which had `epad=2`
at +3.1% *slower*. Two measurements disagreeing is what forced the repeat; had
they happened to agree, a 19% win would have gone into this report on a single
sample. **GUARDRAILS §6 asks for warm-up hygiene; this says the sample count is
the other half of it, and that a result which reverses a prior conclusion should
raise the bar rather than lower it.**

### Which epad values were ever actually compared

Worth stating plainly, because it changes what the shipped constant means:

| | epad values measured | on |
|---|---|---|
| desktop | `0` and `8` only | RTX 4000 Ada |
| here | `0, 2, 4, 8, 16` | Radeon 8060S |

`flashepad`'s `? 8 : 0` came from a **two-point** comparison. `epad = 4` — which
wins outright at four of five head dimensions here, and beats `8` even where the
shipped rule already pads — was never in the running on either machine. That is
not a claim it wins on NVIDIA; it is a claim that nobody has looked.

**The experiment to run on the desktop** is the same five-point `epad` sweep at
the same five head dimensions. Two outcomes, and neither needs a vendor branch,
which `CLAUDE.md` forbids in `src/`:

- if `4` also wins or ties on NVIDIA, the constant simply changes for everyone;
- if it does not, then the pad genuinely depends on the device and the rule has to
  be derived from a **measured property** — the natural LDS access width, which is
  what the alignment table above says is actually doing the work — added to
  `Device` alongside `tile` and `sharedbudget`. Never from a vendor name.

### The sweep does not reproduce even at its own shape

The encoder runs the flash kernel at **five** distinct shapes, not one, and the
dispatch names carry the workgroup counts so they separate cleanly. Median of 5
interleaved samples each:

| shape (`groups`) | implied | ms | →`epad=2` | →`epad=4` | →`epad=8` | spread |
|---|---|---|---|---|---|---|
| `(64, 8, 1)` | Lq=4096, H=8, B=1 — **the sweep's shape** | 48.17 | -0.5% | +6.3% | +9.5% | 11–17% |
| `(4, 8, 16)` | Lq=256, windowed | 42.93 | -0.4% | +0.5% | +0.7% | 3–6% |
| `(1, 2, 1024)` | Lq=64, 2048 head-batches | 4.60 | +0.6% | -1.2% | -1.7% | 7–12% |
| `(1, 16, 16)` | Lq=64, 256 head-batches | 1.87 | -8.7% | -11.5% | -5.5% | 12–23% |
| `(1, 4, 1024)` | Lq=64, 4096 head-batches | 1.25 | -0.3% | +0.1% | +0.4% | 2–6% |

`groups=(64, 8, 1)` at BR=64 is Lq=4096, H=8, B=1 — **exactly** the shape the
sweep and the NVIDIA baseline are both quoted on. Compare the two rankings for
that one shape:

| | `epad=0` | `epad=2` | `epad=4` | `epad=8` |
|---|---|---|---|---|
| sweep, kernel alone | 14.807 | 19.254 **(+30%)** | 13.867 (-6.4%) | 14.468 (-2.3%) |
| same shape, in the model | 48.17 | 47.95 **(-0.5%)** | 51.23 (+6.3%) | 52.77 (+9.5%) |

**The orderings disagree completely, and the sweep's single largest effect —
`epad = 2` costing +30% — is simply absent.** A +30% penalty on 48 ms would be
+14 ms; the measurement is -0.2 ms against a spread of 16%. Whatever else is
uncertain here, that effect does not exist in situ.

So this is not the usual "the microbenchmark used an unrepresentative shape". The
shape *is* representative — it is half the encoder's flash time — and the
benchmark still fails to predict the same kernel at the same shape inside the
model.

**Not claimed, worth testing next:** the sweep allocates its own `q/k/v` fresh,
while the model's tensors come out of the slab allocator at whatever offset the
graph gave them. Since the entire effect under investigation is the *alignment*
of a shared-memory row, a difference in the base alignment of the global-memory
operands feeding it is the first mechanism to look at. That would make the
sweep's numbers an artefact of how the sweep allocates — a property of the
harness rather than of RDNA.

### Phase 3.4, settled

- The shipped `flashepad` / `flashrpad` defaults are **kept** on this machine.
- The sweep says they are wrong by 22–32%. The model says the setting does not
  matter: at the whole-encode level, at the per-kernel level, and at the sweep's
  own shape. The model wins.
- The desktop's decision to ship `rpad = 0` on the strength of a real-model check
  rather than its sweep is independently confirmed here, from a machine where the
  sweep pointed the other way and by a larger margin.
- The one thing that would change this is the alignment hypothesis above, which is
  a question about the benchmark harness, not about RDNA.

## 2026-08-03 — Depth Anything V2: the gap is one redundant copy, not convolution

Run through `DepthAnythingRunner` itself (frozen kernels **54 hits / 0 misses**),
25 reps, `IQR/median 0.9%`:

| | |
|---|---|
| forward | min 934.0, **median 976.1 ms** |
| PyTorch, desktop card | 24.7 ms |
| this port, desktop card | ~380 ms |

### Where it goes, and it is not where the record says

| family | ms | share | dispatches |
|---|---|---|---|
| **elementwise (ndmap)** | **785.12** | **83.2%** | 31 |
| gemm | 79.61 | 8.4% | 50 |
| **convolution** | **44.09** | **4.7%** | 31 |
| elementwise (broadcast) | 19.35 | 2.0% | 203 |
| softmax | 15.28 | 1.6% | 12 |

`small-models/REPORT.md` says the miss is "convolution throughput". **Convolution
is 4.7%.** That claim needs re-testing on the 3070 before it is inherited again;
it may be true there and false here, but it is stated unconditionally.

### It is a single `clone`

```
ka f=#gpu_ndmap_flat! groups=(8220, 1, 1)     726.57 ms   x12     ← 77% of the model
```

`8220 = 6 heads x 1370 tokens`, 12 dispatches = 12 DINOv2 blocks. Of the ops
writing >5M-element tensors twelve times — `bmm`, `_softmax`, `clone` — the first
two are timed separately, so this is **`clone.default` on the `[1, 6, 1370, 1370]`
attention matrix**: 60.5 ms each, 45 MB in and 45 MB out, i.e. **~1.5 GB/s** on a
card that does hundreds.

The chain is `_softmax -> clone -> expand_3 -> view_7 -> bmm_1`.

**Redundant, not dead — and the difference took a second query to establish.** A
direct reader search said the clone's output had *zero* consumers, which reads as
dead code; the consumer is two view levels down and the first search missed it.
Recorded because "dead" and "redundant" have different fixes and the wrong one
would have been a silent correctness change.

It is redundant because the source is a **transient**, produced by `_softmax`,
with exactly **one** reader, at the **same shape and dtype**, already contiguous.
Nothing aliases it and nothing mutates it. `expand_3` can read `_softmax`
directly.

Two independent defects, either of which is worth more than the whole padding
sweep that preceded it:

1. **The copy should not exist.** Eliding it is the rewrite `foldrelu` and
   `foldbatchnorm` already perform — repoint the consumer at the source. A
   `dropclone` pass belongs in that family, gated on: source is a transient,
   single reader, identical shape/dtype, not a graph output.
   **944 -> ~218 ms, 4.3x**, which moves this port from ~15x off PyTorch to ~3.5x.

2. **The copy is ~100x too slow even if kept.** A contiguous 45 MB copy at
   1.5 GB/s is a kernel problem in its own right, and it is the same
   `gpu_ndmap_flat!` every elementwise op in the library uses. The second-largest
   ndmap (`groups=(175960,1,1)`, 57.57 ms, x12) moves more elements in a
   twelfth of the time, so the pathology is specific to this launch shape —
   8220 groups of 1370 elements — not to `ndmap_flat` generally.

Neither has been fixed here: both are DNNKernels/Lava work on a model this
machine does not own, and (2) needs the desktop to confirm the launch-shape
effect reproduces on NVIDIA before anyone changes the kernel for everyone.

### Method note

The first measurement of this reported `spread 438.5%` from 15 reps — one
outlier, not a distribution. The re-run at 25 reps with quartiles gives
`IQR/median 0.9%` and `max/min 1.1x`. Quote the quartiles.
