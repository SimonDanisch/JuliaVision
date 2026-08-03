# Guardrails

Every `projects/*/BRIEF.md` points here rather than copying this list, so there is
one copy to change.

Each of these exists because it was broken, once, at cost. They are cheap to
follow and each one would have saved hours.

## 1. Rule 0 — the driver is not the suspect

Full text in `dev/Lava/spirv-intrinsics.md`. Short form: Lava is a few months old
and was written fast; the NVIDIA driver is years old and ships to millions. When
a kernel misbehaves the prior is that **the bug is ours**.

On 2026-08-02 a "hardware lane cap above 256 threads" that had stood for months,
with `spirv-val` passing and driver-reported register counts as evidence, turned
out to be one line of ours — `hash(spirv_bytes)` samples a large `Vector`, so two
modules differing at one byte collided in our pipeline cache.

Before the word "driver" is used: `spirv-val --target-env vulkan1.3`;
`Lava.enable_gpu_av`; hunt UB in our own output; and above all **write the same
kernel in GLSL, compile it with `glslangValidator`, run that module through the
same dispatch.** If glslang's is correct and ours is not, the bug is ours.

## 2. No new global toggles

`kernel-library-review.md` finding 3 counts **34 mutable toggles in DNNKernels
and 84 `Ref`s in Lava**, and the target is **0 and 1**. That count is the
refactor's progress metric, so adding to it moves the number backwards. Two were
added on 2026-08-02 while the review sat unread.

New behaviour goes on a plan object or a `Ctx` field, not in a module-scope `Ref`.

## 3. Assert output coverage, not just values

A kernel that skips work looks like a speed-up. That is literally how the
workgroup bug entered: a "3.4x" permuted copy had written a quarter of its
destination. Assert that every element was written, not only that the written
ones are right.

## 4. Assert pipeline count in any A/B

```julia
n = length(Lava.PIPELINE_CACHE)
# ... compile the variants ...
@assert length(Lava.PIPELINE_CACHE) == n + nvariants
```

One line, and it catches the whole cache-collision class: two "different"
variants that silently ran the same shader would otherwise measure identical and
be reported as "no difference".

Twice in two days now. After the `hash(spirv_bytes)` collision came
`clear_kernel_cache!`, which emptied `LINKED_KERNEL_CACHE` but not
`LAUNCH_PLAN_CACHE` — and the latter holds its own `VkPipeline` and is consulted
*first*, so the function silently did nothing and an A/B reported "no difference"
for **six variants, one of which had its `OpStore` deleted**. Fixed in Lava
`740d982`. The shared signature is *an A/B that finds no difference between
things that must differ*; treat that outcome as a broken harness until this
assertion says otherwise.

## 5. Never rank an item against an unmeasured denominator

The fp8 item was ranked at "≤2x on 44 ms" against a ceiling measured for
**fp16→fp32**. Whether `K32` doubles throughput was never checked — and `int8` is
also `K32` and already wired, so it can be checked with no new code. Measure the
denominator before the item is allowed to start.

## 6. Measurement hygiene

`plans/perf-plan.md` ends with twenty-two ways a measurement here produced a
confident wrong number. The three that recur:

- Gate the warm-up on the **SM clock** (`nvidia-smi --query-gpu=clocks.sm`), and
  print the clock next to the numbers. This card idles at 210 MHz of 2265.
- Interleave A/B **in both orders** — whatever runs second carries a bias.
- Cross-session variance is ~13%. Only same-session interleaved numbers are
  evidence, and **only the desktop's numbers go in `perf-plan.md`**.

## 7. Read the plan before proposing work

`plans/kernels-to-port.md`, `plans/models-to-port.md`,
`plans/kernel-library-review.md` and `dev/Lava/spirv-intrinsics.md` are the
inputs. A work plan produced without reading all four has been produced twice and
was wrong both times — most recently by proposing five parallel branches that
would all have collided with a refactor described in a file that had not been
opened.

## 8. No device-owned handle in a module-level cache

A `VkPipeline`, `VkShaderModule`, `VkPipelineLayout` or `VkDescriptorSetLayout`
belongs to the `VkDevice` that created it. Binding one from device A into a
command buffer on device B is undefined behaviour.

Four caches in Lava hold exactly those handles at module scope, keyed without the
device — `PIPELINE_CACHE`, `LINKED_KERNEL_CACHE`, `LAUNCH_PLAN_CACHE`,
`GFX_PIPELINE_CACHE`. Two devices running the same kernel produce the same key and
the second one gets the first one's pipeline. This is the same class as the
`hash(spirv_bytes)` collision fixed on 2026-08-02: a cache keyed on too little,
returning an object that does not belong to the caller.

Device state goes on `VkContext`. If you are adding a cache and it holds anything
the driver created, it goes on the context, not in a `const`.

**Done, 2026-08-02, and the intermediate step is the warning.** Twelve of these
were first fixed by keying the `const` dicts on `ctx.id`. That made two devices
work and the probe found seven real defects — but it satisfied the letter of this
rule while breaking its point: entries outlived the device they described,
`ctx.id` was a surrogate for "the object this should have been stored on", and
`RESET_CALLBACKS` existed mostly to empty them. Two of the twelve
(`BLIT_PIPELINE`, `TIMESTAMP_POOL`) were never keyed at all and stayed broken on
a second device the whole time, because the probe's path never reached graphics
or dispatch profiling.

They are now concrete fields in `DeviceCaches`. The obstacle had been include
order — `VkContext` is defined before the cached value types, so a field would
have to be `Any` and cost inference on a per-dispatch lookup — and the answer was
to move the nine *type definitions* ahead of it (`runtime/coretypes.jl`), not to
accept the surrogate key. Methods can stay where they are; only types are
ordered.

So: keying by device is not the same as owning per device, and "the cache is
keyed correctly" is not the test. The test is whether the state dies with the
device.

## 9. Commit and report at the end of every step, not at the end of the project

Append findings to `plans/projects/<name>/REPORT.md` and push. The next agent may
be on another machine, or three weeks later. The in-session task list and any
agent memory do **not** travel; the repo does.

**Per step, not per project.** On 2026-08-02 all three running agents were killed
at once, ~80 minutes before anyone noticed. `kernels-refactor` had committed each
step as it finished and lost nothing. `lava-core` was holding a completed
Rule-0 GLSL differential and a 234× accuracy fix as dirty files and came within
one crash of losing both. Neither had written a line here, so both reports had to
be reconstructed by hand from commits and worktrees.

A step that is finished and uncommitted is not finished. If the work is not yet
worth a commit on the project branch, it is not yet a step.

**A running agent card is not evidence the agent is alive.** Those three kept
showing an incrementing timer and a stop button for well over an hour after they
died. What is evidence: the last write time in the worktree
(`find <tree> -type f -newermt '-20 minutes'`) and `git log` on the branch.

Record negative results with the same care as positive ones — roughly half the
value in `perf-plan.md` is knowing which six things were measured and lost.
