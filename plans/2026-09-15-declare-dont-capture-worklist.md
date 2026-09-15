# Declare, don't capture — worklist

Baseline and acceptance criteria: `2026-09-15-declare-dont-capture-baseline.md`.

The shape: DNNKernels declares the whole graph to Mantle from the ATen op list —
a `Transient.Buffer` per intermediate, a `compute!` pass per op with
`use(p, in; read = true)` / `use(p, out; write = true)`, dispatches declared over
those handles — then `Plan` runs its seven phases as designed and `run!` walks or
replays. Mantle owns every byte. Ops are not rewritten: their bodies emit instead
of launching, which is a one-line change per launch.

## Done — deleted

**Mantle core.** `RECORDING_PASS`, `recordingpass`, `record_into` (both forms),
`LaunchPasses`, `capture_launch!`, `capture_call!`, `recordarg`, `usagekey`,
`useleaves!`, `usefields!`, `Rebound` — 242 lines from `runtime/dispatch.jl`. The
three interception sites: the launch diversion in `vulkan/array/ka_backend.jl`,
the kernel detour for a recorded copy in `vulkan/array/gpuarrays.jl`, and the
host-transfer guard in `vulkan/runtime/memory.jl`. `usagekey(::LavaArray)`, the
`:usagekey` vocabulary entry, and the `record_into`/`recordingpass` exports.
`d2dcopy_kernel!` and `fill_kernel!` STAY — a graph declares them.

**Mantle tests.** `test/vulkan/test_captured_launch_fidelity.jl` deleted with its
include. `test_partitioned_recording.jl`'s three `record_into` graph builds
rewritten as `compute!`/`dispatch!`/`use`; its two assertions that a host
read/write inside a capture throws are gone with the guard. `test_rocm.jl`'s
"record_into sees more than kernel launches" deleted.

**ROCm extension.** `recordhook`, four `recordop!` methods, `HIPLaunch`,
`BLASMul`, `retain`, `usagekey`, the `RECORD_HOOK` install — 158 lines.

**AMDGPU-local.** `RECORD_HOOK`, `recordhook` and all five consult sites. The
local patch set is 109 lines across four files, down from ~200 across eight, and
`AMDGPU.jl`, `array.jl`, `ROCKernels.jl` and `runtime/hip-execution.jl` are back
to upstream. What remains is unrelated to any of this and must stay: the fp16
`generic_matmatmul!` branch, `gemmEx!`, `begincapture!`/`endcapture!` for HIP
graphs, and the scoped `stream!` fix.

**DNNKernels.** `src/workspace.jl` and `src/mantle.jl` deleted whole (106 + 823
lines): `Workspace`/`scratch!`/`reset!`, and the dead declarative chain plus
`MantlePlan`/`place`/`Discovery`/`rediscover`/`declarations`. `plan.jl` cut from
331 to 114: `Slab`, `place(::Slab, …)`, `planslab`, `checkslab`, `lifetimes`.

Kept from `plan.jl`, because emit needs exactly these: `escaping`, `rootbuffer`,
`SHAPEONLY_VIEWS`, `materialisedview`.

## Two findings that must not be lost

**The export's `b.live` is not safe to place against.** `lifetimes`' docstring:
a buffer stays reachable long after torch considers it dead, so reusing its bytes
at the annotated point corrupts the result. Mantle derives liveness from declared
use, which is the same walk done once by the party that knows. `b.live` is a
cross-check, never an input.

**A shape-only view of a permute forces a copy.** `materialisedview` decides it
from the graph, and names exactly the 51 buffers totalling 290.2 MB that
`contiguous` allocates at run time on SAM 2's encoder. Declared, each is a
transient and a permute pass.

## Remaining — deletions

1. `execute.jl`: the `mgraph === nothing` branch; `emit`'s lazy `Broadcasted`
   return; `dest`/`alloc`/`tupledest`'s slab paths; `makeview`; `contiguous`;
   `value`'s view materialisation.
2. `driver.jl`: `scratchfor`, `recordplan`, `RecordedPlan`, the `record` flag.
3. `context.jl`: `Ctx`'s `slab`, `plan`, `ws`, `lazy`, `rec` fields.
4. `fuse.jl`: `fusableset`'s lazy set and its `duplicate` guard. The GROUPING
   analysis stays and becomes group codegen.
5. `kernels/extern/attention.jl`: `stridedroot`, `ondevice`.
6. `SAM2Runner/src/sam2.jl`: the `record_into` call in `handover`.
7. `tools/bench_masked_flash.jl`, `tools/bench_q8_tiles.jl`: 3 uses.

## Remaining — conversions, not deletions

About 30 `scratch!(ctx, …)` sites become declared transients, concentrated in
`attention.jl` (12), `matmul.jl` (7), `conv*.jl` (6), `flash.jl` (2),
`maskedprefill.jl` (3), plus 17 loose `KA.allocate` and 7 `similar`. Each becomes
`Transient.Buffer(ec.g, T, dims...)` declared by the op that needs it — an op's
internal scratch is in no ATen graph because the export is at torch granularity,
and emit is the lowering that puts it in the Mantle graph.

Then `emitop!(ec, op, ::Val{aten})` per op, 88 methods, same dispatch as
`runop!`. Patterns and pseudocode for each: see the plan in the session
transcript — one kernel one output, host scalars, multi-output via `tupledest`,
internal scratch, multi-launch with one pass per dependent stage (`unordered =
true` for the atomic split-K accumulation, which capture could never express),
library call as a `Dispatch` over a callable, `fill!`/`copyto!` as core's two
kernels, fused group as ONE dispatch from `groupkernel(op.expr)`.

## Done — prerequisite landed and verified

N-D transients: `TransientBuffer{T,N}` carrying `dims`, `Transient.Buffer(g, T,
dims...)`, and `materialize!` for the KA backends moved into core over one new
verb, `deviceslice(dev, T, dims, mem, offset)` — three backend bodies that each
hard-coded `(t.n,)` became one line each. Needed because `use` interns by object
identity, so a reshaped view cannot be the thing named: the handle must be the
transient, so the transient must carry the shape. Verified on Vulkan and Host,
55 arena assertions, ROCm 54, all eight philosophy guards.

## Correction: a callable kernel is NOT generally declarable

Claimed earlier, on the strength of the ROCm extension, that a library call is
just "a `Dispatch` whose kernel is a callable, which `compile_dispatch` already
handles". True of the ROCm extension — `obj isa KA.Kernel || return
ROCmCallDispatch(...)` — and **false of Lava**, whose `compile_dispatch` goes
straight to `get_or_build_iter_plan(::KA.Kernel{LavaBackend}, …)` with no
fallback. Measured by trying it: `MethodError: no method matching
get_or_build_iter_plan(::Elementwise{typeof(identity)}, …)`.

The deeper problem is not the missing method. A callable whose body performs an
ordinary launch worked only BECAUSE of capture: on HIP the launch went to the
stream being captured, so it landed in the graph. With capture gone it is an
immediate submit from inside a pass — the unreplayable thing. A callable dispatch
is sound only where the backend's submission mechanism swallows it (a HIP
capture, rocBLAS on a captured stream) and is not a general answer.

Consequences:

1. Elementwise ops cannot be a callable that broadcasts. They need a real
   `@kernel` parameterised on `f` and the operand count, with per-operand stride
   descriptors so a `(C, 1, 1)` bias meets a `(W, H, C)` activation without
   materialising. That kernel IS the group kernel the elementwise-fusion plan
   wants, so it is not a detour: write it once, use it for every elementwise op
   and for `fused.elementwise`, and the baseline's 1014 broadcast dispatches
   become the measurement.
2. Library calls stay callable dispatches, but that is a ROCm capability — Lava's
   GEMM is our own kernel and needs no equivalent. Worth stating in
   `compile_dispatch`'s docstring on both sides rather than leaving it an
   accident.

## Also landed

`Mantle.ResourceView` + `viewof` + `elementrange`, and `use(p, v::ResourceView)`
claiming the parent scoped to the view's elements — what replaces `makeview` →
wrapper stack → `stridedroot`.

It surfaced a latent core bug: `Liveness` resets every transient interval and
recomputes from usage ids, but a `range`d usage is interned as the `BufferRange`,
so `touch!`'s record was discarded and the parent got NO interval — "transient 1
is never used by any pass" while two passes were writing it. Latent because
`range` had only been used on persistent buffers, which are not placed. Fixed
with `transientfor`, resolving a usage id to its parent transient through the
slice.

`src/emit.jl` exists: `EmitCtx`, `emitgraph`, `declare!`/`make` (transient unless
escaping), `viewfor` (shape-only views by construction, a loud error otherwise),
`operand`, `dest`/`dests`, `scratch`, and an `emitop!` fallback naming the
unported op. Five ops ported (`mul`, `div`, `add`, `clone`, `leaky_relu`), all
five needing a rewrite onto the real elementwise kernel per above.

Verified: the scaffolding declares a hand-built two-op ATen graph correctly — `t`
a transient, `x`/`y` owned buffers, `tv` a `ResourceView` over `t`, two passes —
and `Plan` reaches `Pipelines` before hitting the callable problem. Note
`evalshape` reverses torch order, so a declared `(4, 2)` is `(2, 4)` in Julia.

## Ecosystem: do not implement the KI contract for ROCm or Metal

It is being implemented upstream right now, and both are in flight as drafts:

  * AMDGPU.jl PR #1047 `KernelInterface`, open/draft, 8 files +327/-125, branch
    `christiangnrd:interface`, head `4d75db5d Support KI 0.2`
  * Metal.jl PR #917 `KernelInterface`, open/draft, 7 files +295/-117, branch
    `JuliaGPU:kainterface`, head `a00d6cfe Adapt to KI 0.2`

Both say "Do not merge until KernelInterface has been reviewed and interface
fully decided", and KernelAbstractions has merged the groundwork (#733 rework
into a sibling package, #754 KI 0.2 element-typed indexing queries, #742 auto
launch sizes, #751 require `adapt(::Backend, x)`). So the interface is still
moving; writing against it is fine, reimplementing it is not.

Branched off both, 2026-09-15:

  * `dev/AMDGPU` on `mantle/kainterface` from `cgnrd/interface`. Our four
    remaining patches (fp16 `generic_matmatmul!`, `gemmEx!`, HIP
    `begincapture!`/`endcapture!`, scoped `stream!`) apply CLEANLY — 109 lines,
    no conflicts, so none of them collides with the interface work.
  * `dev/Metal` on `mantle/kainterface` from `cgnrd/kainterface`. Nothing of ours
    to carry.

`dev/AMDGPU-local` (the 2.2.1 snapshot with no remote) is what the Manifest
still points at, so switching to the branch is a `Pkg.develop` and is the user's
call. Metal is not dev'd and has no hardware here.

Only the HOST backend's KI contract is ours to write.

## Revised step 0: one record, and `dispatch!` takes a function

`dispatch!` should take a plain Julia function and Mantle should build whatever
the backend needs. That deletes verbs rather than adding them:

  * `kernelfor(k, group, backend) = k(backend, group)` exists only to rebuild the
    `@kernel` macro's object per backend. With a function it is
    `KI.kernel_function(backend, f)` — which is KI's own per-backend hook, so
    Mantle should CALL it, not alias it. Delete `kernelfor`.
  * `callgroup` exists only because `@kernel` encodes the workgroup size in the
    TYPE (`StaticSize`/`DynamicSize`). Its own docstring says so. KI takes
    `workgroupsize` as a launch argument, and `Dispatch` already carries `group`
    as a field, so there is nothing to negotiate. Delete `callgroup`.
  * That also removes the CAUSE of the 29% autotune bug fixed this morning:
    `compile_dispatch` asked `KA.launch_config` whether a group had been
    requested, and it substitutes its own default for a `DynamicSize` kernel so
    the answer was never `nothing`. No type-level encoding, no substituted
    default, nothing to misread.

Migration is NOT simply incremental, and this is the thing to decide first:
`@kernel` generates a FUNCTION. `foo!` from the macro is a `Function` with a
method taking a backend and returning a `KA.Kernel`; a macro-free kernel is a
`Function` with a method taking arrays. `d.kernel isa Function` cannot tell them
apart, so `compile_dispatch` cannot route on it.

Three ways out, in order of how much I like them:

  1. **Port in one pass.** `@kernel` never reaches `dispatch!`, `Dispatch.kernel`
     is always a plain kernel function, no coexistence and no discriminator.
     Touches every kernel in Mantle, Hikari and DNNKernels at once.
  2. **`hasmethod(f, Tuple{typeof(backend)})`** as the discriminator — true for a
     macro-generated constructor by construction, false for a kernel body. Not
     fragile in practice, and it buys a migration window where both forms work.
     Costs a structural test on a naming convention.
  3. Wrap the new form at the call site (`dispatch!(p, KI.kernel(f), …)`), which
     is a difference at the call site and therefore the thing we said we did not
     want.

`callgroup` can NOT go first, though — it returns `nothing` for a `StaticSize`
kernel deliberately, because "passing it again keys a second, identical iteration
plan for the same launch", so while any `@kernel` constructor still reaches
`dispatch!` it is doing a real job. It goes WITH the KA path, last. Order:

  1. Lava and host `compile_dispatch` accept a plain function via
     `KI.kernel_function`. Testable immediately; unblocks writing kernels
     macro-free.
  2. Port the kernels — Mantle's own, then DNNKernels' 88 ops.
  3. Then delete the KA path, `callgroup`, `kernelfor`, and `:callgroup` from the
     vocabulary, together.

`Dispatch`/`Trace`/`LibCall` collapse into ONE record whose first field is
whatever is to be called, with the backend's `compile_dispatch` deciding by its
type — `typeof(mul!)` is a singleton, so a library op is a specialisation and
needs no marker, and a missing ndrange is the natural tell for the library shape.
`Trace`'s collapse is deferred behind the DNN path: same shape, but ray tracing
has thinner coverage on this machine and is not blocking.

## Settled empirically: kernels go macro-free, and ROCm already takes them

The `@kernel`-vs-KI question is no longer a judgement call. The ROCm extension
now compiles a plain function through `KernelInterface`:

    kifunction(f, backend) = !hasmethod(f, Tuple{typeof(backend)})
    kifunction(d.kernel, backend(dev)) && return kicompile(c, dev, d)

`kicompile` is `KI.kernel_function` (the compile — `hipfunction`) plus
`KI.auto_launch_sizes` (the launch configuration), both at `Pipelines` time so a
capture contains no host work. `ROCmCompiledDispatch.ctx === nothing` marks the
KI form, since a plain function has no leading `CompilerMetadata`; the call
branches on a field whose type is known, so it specialises away.

**`auto_launch_sizes` is the autotune `hipcompile` did by hand, done right.**
`kernel_max_work_group_size` for the occupancy limit, `threads_to_workgroupsize`
to SHAPE it to the ndrange, then `cld.(ndrange, wg)`. The shaping step is the one
the hand version skipped — the 29% encoder bug. Every kernel that moves to KI
becomes unable to have it, and `hipcompile`'s 40 lines die with the last
`@kernel`.

Verified: ROCm 49/49 on `@kernel` unchanged, 4/4 on a macro-free KI kernel
through the same `dispatch!`.

Two facts that only appeared by running it, both of which paper reasoning got
wrong:

  * `AMDGPU.ROCBackend` (= `ROCKernels.ROCBackend`, KA) and
    `AMDGPU.ROCInterface.ROCBackend` (KI) are DIFFERENT TYPES on that branch, so
    the KI path names the second explicitly. Lava has both interfaces on one
    `LavaBackend`; ROCm does not.
  * `KI.get_global_id()` returns `@NamedTuple{x::Int, y::Int, z::Int}` and is
    1-based, and there is NO implicit ndrange bounds check the way `@kernel` had.
    A kernel writing `i = KI.get_global_id()` then `i > length(dst)` fails to
    compile with `jl_f_throw_methoderror`, which is at least the right error.

## Landed: the elementwise / group kernel

`DNNKernels/src/kernels/elementwise.jl` — macro-free, arity-specialised
(`ew1!`/`ew2!`/`ew3!`) over a shared `bcindex`, with `bcstrides(od, id)` giving an
operand ZERO stride on every axis it is broadcast along. So a `(1, 1, C)` bias
meets a `(W, H, C)` activation in one dispatch with no materialised copy — which
is what replaces the 1,014 pairwise broadcast expansions in the baseline.

Arity-specialised rather than variadic over a tuple of operands, because
`Mantle.resolve` is applied per element of a dispatch's argument tuple: a
tuple-of-resources argument would arrive as unresolved handles, and each operand
has to be a top-level argument anyway for `use(p, x; read = true)` to name it.

Verified on ROCm: bias-broadcast plus ReLU, values exact against the host
reference, and `peakbytes(pl) == prod(od) * sizeof(Float32)` — **Mantle placed
the intermediate**, which is the baseline's zero becoming real.

## Landed: NeuralLUT runs entirely declared, on both backends

All 7 of its atens are `emitop!` methods, the runner builds a Mantle graph and
replays a recorded plan, and the result matches an independent host reference to
**4.4e-6** max absolute error on Lava and the same per-op tolerance on ROCm.
26 atens become 35 passes; the placer takes the peak from 6.96 MiB unaliased to
2.0 MiB.

`NeuralLUTRunner/test/test_declared.jl` is the gate, and
`DNNKernels/test/hostref.jl` is the reference: a host interpreter in plain Julia
for the 7 ops, four nested loops for the convolution. Against the DEFINITION and
not against `runop!`, because `planslab`, `Workspace` and the lazy path are gone
and there is no A/B left to run.

New kernels, all macro-free: `ew1!`/`ew2!`/`ew3!` (`kernels/elementwise.jl`),
`tilecopy!`/`sumdims!` (`kernels/shapeops.jl`), `bnstats!`/`bnapply!`
(`kernels/batchnorm.jl`), and `conv2d_igemm_ki!` (`kernels/conv_igemm.jl`), the
175-line ggml-derived implicit GEMM ported off `@kernel`.

### What the port had to fix underneath it

Five things, four of them upstream, and each was silently wrong rather than
absent:

  * **`KI.localmemory` had no call-site id.** A backend keys its workgroup
    global on what it is handed, so two buffers of one type and shape were ONE
    buffer, with the second write landing on the first tile. That is exactly the
    igemm kernel's `Ash`/`Bsh`, and it is why Lava had declined to implement the
    function at all. The id is in KI's signature now
    (`localmemory(T, dims, [id])`), AMDGPU/Metal/POCL/OpenCL thread it into the
    global's name, `KA.SharedMemory` forwards it instead of dropping it, KA's
    own `performant_matmul` example is fixed, and Lava implements it in one
    line. `test_declared_kernel.jl`'s `twotiles!` pins it.
  * **The ROCm extension flattened a dispatch's launch geometry** to
    `prod(gridsize)`. Correct for a `@kernel`, whose index comes out of the
    `CompilerMetadata`; wrong for a macro-free kernel that asks the hardware, so
    the convolution's second grid axis collapsed and every workgroup read block
    1. `ROCmCompiledDispatch` carries the tuples.
  * **`MantleROCmExt` was never registered as an extension.** No `[weakdeps]`,
    no `[extensions]`; every session that used the ROCm backend had `include`d
    the file by hand into `Main`. It loads on `import AMDGPU` now.
  * **`buildskernel`/`kikernel`/`kibackend`,** in core. The KA-vs-macro-free
    decision was made three times and the three did not agree: Lava asked
    `hasmethod` in `compile_dispatch`, the extension asked it as `kifunction`,
    and core's `bake` did not ask at all, so the host produced
    `MethodError: no method matching ew2!(::CPU)`. `kibackend` exists because a
    device can have two backend objects and only the backend knows which one KI
    speaks to.
  * **`viewof` nested instead of composing,** so an ATen reshape-of-a-reshape
    (`view_9` of `view_8`) produced a `ResourceView` whose parent was a
    `ResourceView`, and `use` cannot forward a `range` to one. It composes now,
    and bounds-checks against the parent's extent while it is there.

Also: `size(::Buffer, d)`, which Base gives an `AbstractArray` for free and a
`Buffer` is not one.

### Still `@kernel`, still to port

`conv_implicit.jl`'s `conv2d_igemm!` is the old spelling of the kernel that was
ported; it stays only because `runop!` still reaches it through
`convolution!`, and both go when `ops.jl` does. The coopmat and direct
convolution paths, the transposed convolution, and the 1-D/3-D cases are
unported: `emitop!` names each one rather than running it as something else.

## The CPU path is Lava on lavapipe, not a KI backend for `KA.CPU`

`KernelInterface` has no backend for `KernelAbstractions.CPU`, so `kikernel`
refuses a macro-free kernel on Mantle's host device and
`test_declared_kernel.jl`'s host arm pins that refusal by name.

That is not a gap to close upstream. Stage 1 of the verification order wants
"the same source, on the CPU, layer by layer against the reference", and the way
to get it is **lavapipe**: Mesa's software Vulkan device runs the same SPIR-V
module Lava emits for a GPU, so the CPU run exercises the code under test rather
than a second implementation of it. `KA.CPU` would give a different kernel from
a different compiler, which is the thing that makes a CPU-vs-GPU comparison
worth less than it looks. Keeping `HostAPI` for graphs that need no kernels is
still useful; it is not the parity path.

## Landed: a library call is a declared pass member

`dispatch!(p, mul!, (use(p, C; write = true), use(p, A; read = true),
use(p, B; read = true)))` — three arguments, no ndrange, and the absence of the
ndrange IS the declaration. An ndrange is what Mantle needs in order to divide
work into workgroups; a `mul!` has none, because the thing on the other side
decides its own launch. So no second verb name and no trait: the caller hands
over the function it wants called.

**This is the answer to "a `mul!` can hide a very complex GPU dispatch behind
rocBLAS".** It costs no boilerplate per operation, because `mul!` on a device
array ALREADY routes correctly in each ecosystem: measured here, the same call
reaches rocBLAS through AMDGPU.jl and this backend's cooperative-matrix GEMM
through Lava's own `mul!`, both exact. What the caller still has to state is the
direction of each argument, which is the one thing no library announces and the
thing `useleaves!` could never know. `fft!`, a `cudnn` convolution and a `sort!`
arrive by the same route.

`Mantle.Call` is core's and holds the function with its resolved arguments;
`iscall(d)` is the predicate; `ROCmCallDispatch` is deleted, having been this
with an `ndrange` keyword bolted on.

### `runscalls(device)`, and why the answers differ

Asked at COMPILE, so a graph declaring a call on a backend that cannot run one
is refused while it is being built rather than when it is submitted. Core owns
the refusal and the message; a backend only answers the capability.

  * **ROCm: yes.** Its recording is a stream capture, so the rocBLAS kernels the
    call submits land in the capture and the replay is the library's work with
    the library no longer involved. Verified: recorded, replayed from zeroed
    outputs, exact.
  * **Host: yes,** because it WALKS its plans. There is no command buffer for
    the call's work to be missing from.
  * **Vulkan: no.** `run!` submits a recording it built as a command buffer and
    a Lava plan has no walk path at all (`execute!` throws on an unrecorded
    plan), so a host call would run once at record time, submit outside the
    buffer, and be absent from every replay. Silently. The refusal names
    `coopmat_gemm!` as what to declare instead, which is the per-operation verb
    — one backend's cost rather than everyone's.

### The crash this found

`rocblas_create_handle` inside a HIP capture is a **segfault** in libamdhip64,
not an invalidated capture: rocBLAS creates its handle on first use and handle
creation allocates on the capturing stream. So `openrecording` now runs every
call in the plan ONCE before `begincapture!`, against the arguments the plan
resolved, so the library initialises and takes its workspace outside the
capture. One extra execution per call per `record!`, into plan-owned bytes the
replay writes again. This is the backend's business and not a caller's
precondition, because the alternative is a crash rather than an error.

## Also fixed while here

`test/rocm/test_rocm.jl` was `include`ing the extension file into `Main`, which
was the workaround for it not being registered. With the extension declared that
include makes a SECOND `MantleROCmExt` with a second `ROCmDevice` type, and
`Mantle.openrecording` ends up with two methods for two types that are not the
same type. It asks `Base.get_extension` now.
