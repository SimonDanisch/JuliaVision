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
