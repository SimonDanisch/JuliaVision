# DNNKernels on Mantle

One project, landed together, then tested together. Not five phases — the phasing
in the original sketch existed to keep the DNN from being migrated twice, and
once `Update`'s rename stopped being a blocker (see "The rename question,
settled" below) there is nothing left for the ordering to protect.

Two packages change: `dev/Mantle` and `dev/JuliaVision/DNNKernels`.

## Step 0 — environment, before any code

`Manifest.toml:2196` resolves Mantle to `/sim/Programmieren/VulkanDev/dev/Mantle`
while Lava (`dev/Lava`) and DNNKernels (`dev/JuliaVision/DNNKernels`) resolve
inside MantleDev. The two Mantle checkouts are byte-identical at `eae07f2` today,
so nothing is wrong yet — but an edit to `dev/Mantle` does not load, and the
first hour of this project would otherwise be spent wondering why.

1. Add `Mantle = {path = "dev/Mantle"}` to `Project.toml`'s `[sources]`, re-resolve.
2. Run Mantle's existing suite against **MantleDev's** Lava. Ten files differ
   between `MantleDev/dev/Lava/src` and `VulkanDev/dev/Lava/src`, including
   `runtime/command.jl`, and Mantle's ext was developed against the other one.
   `capture`, `replay!`, `reserve_arg_slabs!` and `pack_args_direct!` are all
   present in MantleDev's copy, so the surface Phase 4 needs exists; whether the
   rest agrees is unverified and is a precondition, not a finding to make later.

Green suite here is the gate for starting. Nothing below is worth debugging on
top of an environment mismatch.

## What already holds — not to be redone

- `place` is a dispatch point. `DNNKernels/src/execute.jl:124` defines
  `place(::Nothing, …)` and `plan.jl:38-61` the `Slab` method; a third method is
  the whole of the integration on that side.
- Mantle runs both SAM 2 graphs bit-exact at 180 MB against `planslab`'s 261 MB.
- Dispatches compile at plan time (`Mantle.run!(::Pipelines, …)`,
  `MantleLavaExt.jl:1832`), so a frame cannot reach the compiler.
- Plans hold references to everything they name, which is the property replay
  needs and raw capture lacks.

## The rename question, settled

The original sketch made this Phase 0 and called it blocking: `Update`'s
whole-buffer route renames — `rename!` (ext:914) swaps `dst.store` for a fresh
allocation off the recycler — and that moves an address a baked recording depends
on. Two routes were proposed: write in place, or have a rename invalidate the
recording.

Neither is necessary, because a dispatch's arguments are already patchable.

`record_dispatch!` (ext:2323) packs arguments into `am.ptr + off` and records
`am.address + off` (ext:2335). The command buffer never contains a resource's
address, only argument memory's. Lava's own capture rests on exactly this:
`command.jl:148-153` bakes the arg slab's *address* into the command buffer as a
push constant, "so a replay reads whatever those bytes hold *now*."

And the layout is static. `Mantle.run!(::Pipelines, …)` walks passes with one
`argcursor` (ext:1834) and gives every draw and dispatch a fixed `argoff`
(ext:1880, 1888); `lp.offsets` fixes each argument's position inside a dispatch's
block. So any argument's address is

    am.address + slotbase + argoff + lp.offsets[i]

with all four terms known at compile time. Patching a renamed pointer is one
store at a precomputed offset — not a re-pack, not a re-record.

What a rename genuinely breaks is narrower: the three places that record a
`VkBuffer` **handle** into the command stream rather than reaching it through
argument memory.

1. **The rename's own copy.** `cmd_copy_buffer!` (ext:924) records the
   destination handle and `pool_offset`, and `fresh` comes off the recycler's
   free list — a different handle frame to frame.
2. **The pass barrier.** `build_pass_barrier` (ext:1216) runs at compile time and
   bakes `st.buf[].buffer` (ext:1235). Under rename that barrier names the store
   the plan was compiled with. Not a fault — a buffer barrier scoped to the wrong
   buffer is a missed dependency, and desktop drivers generally treat the range
   as advisory — but it should not survive into a baked plan, and it is worth a
   pass under sync validation regardless.
3. **Indirect draws.** `vk_draw_indirect_in_pass!` (ext:2312) puts the handle in
   the command. Orthogonal to the DNN path; noted so it is not discovered later.

So `Update` keeps both routes and its semantics are unchanged. The constraint
lands on the baker instead, as items M4 and M5 below.

## Design — Mantle

**M1. Arenas move to the `Device`.** `Device` reserves rather than allocates,
keyed by the existing `arena(t)` (ext:286-287, `Buffers()` / `Images()`). `Slab`
(ext:1511) and `allocate` (ext:1524) become reservations against a device-owned
pool. Buffers and images stay separate arenas for the reason already recorded at
ext:278-282 — Vulkan permits aliasing them but the validation layers report it.

**M2. Coarse composition.** A plan is one item to the device-level placer: size
is its peak, interval is when it is live. Two plans in one process then commit
the max of their peaks, not the sum. This is deliberately coarse — the fine
placement inside a plan is unchanged, and modelling a plan as its constituent
items at device level would be re-solving a problem that is already solved one
level down.

**M3. Real capacity into `Problem`.** ext:1557 currently reads

    Mantle.place(Mantle.Problem(c.items[idx], typemax(Int)))

so the bound is decorative. Feed the device's capacity, and make over-budget an
error at compile with the numbers in it — asked, available, and the largest
items. A placement that silently exceeds the device is an allocation failure
several frames later, attributed to whatever allocated next.

**M4. Renameable resources get a global barrier.** One branch in
`build_pass_barrier`: if the resource is an `Update` target, push a
`_MemoryBarrier2` rather than a `_BufferMemoryBarrier2`. Loses the span scoping
for those resources, which is the correct trade — a handle cannot be baked for
something that moves.

**M5. The baker.** Capture the plan's `record!` through Lava's `capture`
(`command.jl:200`), and per invocation: patch the mutable argument slots, upload
changed inputs, submit.

- **The update pass stays out of the baked buffer.** Record it fresh per
  invocation and submit ahead of the replay. It is a handful of copies against
  hundreds of dispatches, so it costs approximately none of the 16.3 ms.
  `replay!` already supports being interleaved with ordinary recording —
  `command.jl:232-240` exists for precisely that case.
- **A baked recording is pinned to one argument slot**, because `base =
  slotbase(am)` is folded into the recorded address. Headless graphs run one
  slot and a fence, which is what a submit-and-read-the-output workload wants
  anyway. Windowed plans get one baked recording per swapchain image, which is
  the same rule the sketch already committed to for buffers: anything that
  varies per frame becomes a function of the slot index.
- **Headless graphs first.** They have no swapchain image to vary.

**M6. The bake is behind a runtime flag.** Not to stage the landing — to let the
testing phase run the placement gates with baking off and the bake gates with it
on. A moved rms number otherwise has two candidate causes and no way to separate
them.

## Design — DNNKernels

**D1. Persist discovery.** Cache the discovered read/write declarations beside
the graph, keyed the way frozen kernels are — graph + shapes + version.
Discovery stays the mechanism: it is what gets lazy views (`fuse.jl`,
`lifetimes` at `plan.jl:92`) and fused broadcasts right, and no static analysis
of the graph JSON reproduces it. It stops being a graph execution per process.

**D2. `place` on the Mantle plan.** A third method at the existing dispatch
point:

    place(p::MantlePlan, sl, id::AbstractString, ::Type{T}, dims)

Alignment 256, matching `plan.jl:29`'s `alignup`. The same size guard the `Slab`
method carries (`plan.jl:46`) applies: a caller asking for a different element
type falls back to `rawalloc` rather than overrunning into a neighbour.

**D3. Escaping buffers persistent.** `escaping(graph)` (`plan.jl:68`) already
computes the set — declared outputs plus anything they are views of. Those become
persistent resources rather than transients, because a step chains eight graphs
through one arena and a value that escapes into the next graph must not sit in
memory the next graph will overwrite.

**D4. Dispatches through `CompiledDispatch`.** Which takes Lava's argument pool
off the path entirely, and is the precondition for M5 — an argument written into
a bump-allocated slab that something else can hand out again is not patchable.

## Assumptions

Stated because they are choices, not findings. Correct either and the design
changes locally, not structurally.

- **DNN inputs go through `Update`.** One upload path for everything, so M4
  applies uniformly. If inputs instead land as a persistent `Buffer` written
  directly, M4 only ever affects the windowed path and the encoder is unaffected
  either way.
- ~~**Capacity is device heap minus a stated reserve.**~~ **Settled, and the
  reserve turned out not to be a fraction.** An arena is one allocation, so the
  binding limit is `VkPhysicalDeviceMaintenance3Properties.maxMemoryAllocationSize`
  — **4 GB** on this machine, against a 29 GB budget. `min(maxalloc, budget −
  committed)` needs no invented headroom and refuses the plans that genuinely
  cannot run. Measured on the way: a placement that fits the *advertised* budget
  (27.49 of 27.53 GB) still OOMs, so budget alone was never the right bound.

## Testing phase

Run after the whole changeset lands, in this order — cheap and local first, so a
failure is attributed before the expensive gates run.

**Mantle, self-contained**

- Two plans in one process commit the max of their peaks, not the sum. This is
  the gate for M1 and M2 together: it can only pass if the `Device` owns the
  reservation, since two plans that each allocate cannot share.
- Over-budget fails at compile, loudly, with asked/available/largest in the
  message.
- Both arenas stay separate: nothing places a buffer and an image at overlapping
  offsets in one allocation (ext:278-282).
- Existing Mantle suite green (`test/runtests.jl`, `test/test_window.jl`).
- `bench/validate.jl` on both update routes — it already covers rename
  (line 127) and in-place (line 128) — under sync validation, which also
  exercises M4.

**DNNKernels, self-contained**

- A cold process builds a plan with **zero** graph executions.
- Cached declarations match a fresh discovery exactly.

**Integration, bake off**

- Parity suites green.
- Encoder rms unchanged: 8.89e-5 / 4.04e-4 / 4.15e-3.
- Peak ≤ planslab — expect 180 MB against 261 MB.
- `checkslab` (`plan.jl:315`) kept as an independent check, computed from
  `lifetimes` rather than from Mantle's placement, so it can disagree with the
  placer it is checking.

**Integration, bake on**

- SAM 2 encode host recording 16.3 ms → ~0.
- Bit-exact against the bake-off run.
- `GC.gc(true)` between invocations is safe. Extend `Lava/test/test_capture_gc.jl`
  from a raw capture to a plan — the plan is where the invariant is actually
  owned, since it holds references to everything it names and a raw capture does
  not.

## Expected payoff

~31% peak on the encoder from placement, ~6% wall from baking, and two fault
classes — argument-pool rewind, replay-after-GC — become structurally impossible
rather than fixed.

## Results, measured

| gate | expected | measured |
|---|---|---|
| encoder bit-exact vs DNNKernels, unbaked | 0 | **0** on all six outputs |
| encoder bit-exact vs DNNKernels, **baked** | 0 | **0** on all six outputs |
| encoder peak | 180 vs 261 MB | **180.0 vs 261.25 MB (−31.1%)** |
| buffers `dest` could not place | 0 | **0** |
| cached declarations == fresh discovery | equal | **equal** (uses, order, bytes, wsbytes) |
| host cost of issuing an encode | 16.3 ms → ~0 | **366 ms → 0.091 ms** |
| baked replay stability | identical | **4 replays identical** |
| `GC.gc(true)` between invocations | safe | **safe**, 6 rounds |
| decoder bit-exact, baked and unbaked | 0 | **0** on both outputs |
| decoder peak | — | **16.02 vs 16.08 MB** |
| both graphs sharing one arena | max | **180.0 MB, not the 196.02 MB sum** |
| the pair against `planslab` | — | **180.0 vs 277.33 MB (−35.1%)** |
| cold process, discovery poisoned | never called | **never called**, build in 6.03 s |
| sync validation, six suites | 0 messages | **0**, incl. 1764+640 barrier matrix |

The host figure needs a caveat: 366 ms is the whole unbaked `runplan` call, which
blocks on the GPU — Lava auto-submits every 64 dispatches and the host waits for
argument slots — so it is not the 16.3 ms of pure recording the spec named. What
it shows is that issuing an encode is now off the host path entirely.

## Found while implementing

Five things that were not in the design and are now part of it.

**`barrierspan` omitted `pool_offset`** (ext:757). A `VkBufferMemoryBarrier2`'s
range is relative to the `VkBuffer`, and Lava suballocates — so every scoped
buffer barrier named `offset = 0` on a pool block whose resources begin tens of
megabytes in. Verified directly: three 4 KB buffers shared one `VkBuffer` at pool
offsets 66512448 / 66516544 / 66520640, and all three barriers said 0. Nothing
had broken, because desktop drivers treat the range as advisory and flush at the
stages given, but it is a barrier that does not describe the hazard it was
derived for. Fixed to `pool_offset + offset`, the same sum `inplace!` and
`rename!` already compute. Found while verifying M4, and it matters *for* M4:
widening renameable resources is only meaningful if the others are scoped
correctly.

**Discovery polluted the graph it was meant to describe.** `rediscover` took the
Mantle graph as a parameter, and `build` handed it the one the plan was being
built into. `place` on a `Discovery` creates a `Transient.Buffer` per non-escaping
buffer, so discovery left 788 transients in that graph; `build` then created the
real 788 and declared its passes against those, leaving the discovery set used by
no pass at all. `Liveness` refuses that by design. It only ever fired on the
fresh-discovery path — a cold cache, i.e. the first run on any machine — and
never on the cached path every later run takes, which is the worst possible
distribution for a bug. `rediscover` now owns a throwaway graph and the parameter
is gone.

**Baking ran the graph without invalidating memoised views, and baked the
result.** Recording a `custom!` pass *runs its body*, so baking a graph performs a
full execution — and therefore has to do everything a run does first. `runplan`
drops the memoised views before every run, because one that `contiguous` had to
copy holds the previous run's numbers. `Mantle.bake!` did not, so the capture was
already wrong and the replay reproduced it faithfully forever: 791 of 791 buffers
wrong from the second op, the three computed outputs off by 1.2, 2.1 and 15.3 —
while the three *constant* outputs matched, which made a total failure look like a
partial one. The fix is `bakeplan!` in the extension, which invalidates and then
bakes. Mantle cannot do this itself: it does not know an ATen graph memoises
anything.

Worth noting how it was found, because the first three hypotheses were all wrong
and all plausible — Lava's argument pool wrapping mid-capture (excluded: an
800-pass `custom!` chain bakes correctly), lazy outputs materialised host-side
after aliasing (excluded: the outputs are `kind=external`, placed and non-lazy),
and D4's deferral (excluded by the same 800-pass test). What settled it was
snapshotting every placed buffer after the capture and comparing against a plain
run: the capture *itself* already disagreed, which moved the search off the replay
entirely.

**A pool grow silently invalidates a baked plan.** M1 and M5 interact: plans share
the device arena, so a later, larger plan grows it and `remap!` re-materialises
every tenant — but a baked tenant's recording holds the old addresses and nothing
can rewrite a command buffer. The replay would then read freed storage,
deterministically and quietly. `remap!` now refuses, and says the order that
avoids it: build every plan that shares a device, then bake.

**D4 is deferred, not done.** Every ATen op is a `custom!` pass, because an op is
not one dispatch — a handler may launch a fused broadcast, a multi-launch
attention or a library GEMM, which is the case `custom!` exists for. `Pipelines`
skips custom passes (`p.kind === :custom ? () : p.dispatches`), so they get no
`CompiledDispatch` and no slot in Mantle's `ArgMemory`. Converting them would
mean discovery recording each op's *launch sequence*, and ops with host-side
branches could not convert at all.

The bake does not need it. Lava's `capture` pins the arg pool with
`reserve_arg_slabs!` (command.jl:212-215), which closes the same fault class, and
after M1–M3 nothing in a DNN plan moves — fixed placement, fixed weights, inputs
written in place — which is exactly the precondition `capture` states at
command.jl:141-143. So M5 for headless DNN is capture-once, replay-per-invocation,
and the "patch the mutable argument slots" half is only needed for the windowed
case where the swapchain image varies.

## Risks

- **Environment mismatch (Step 0).** Ten Lava files differ from where Mantle's
  ext was developed. Highest-probability source of confusing early failures, and
  the cheapest to rule out.
- **M2's coarseness.** Modelling a plan as one item is right when plans are
  serial and their peaks dominate. If two plans are live concurrently and each is
  mostly idle memory, this over-reserves. Measurable against the two SAM 2
  graphs; revisit only if it does.
- **M4 under sync validation.** Widening a buffer barrier to a global one should
  never introduce a hazard, but it changes what the validation layers see, and
  the existing barrier code is where fuzzing previously found three bugs that all
  lived in the interaction between analyses (`phases.jl:4-8`).
- **M5's slot pinning.** A baked recording holding one slot's base address is
  correct only if nothing else writes that slot. Mantle owns its `ArgMemory`
  (ext:1032), unlike Lava's bump pool, which is what makes this safe — but it is
  an invariant worth an assertion rather than a comment.
