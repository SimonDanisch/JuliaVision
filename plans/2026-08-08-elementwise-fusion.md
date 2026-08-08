# Elementwise fusion as a graph rewrite

Replaces the runtime-lazy fusion in `fuse.jl` with a pass that rewrites
`op{a} -> op{b} -> op{c}` into one op carrying a `FusedOp`, folds unary
elementwise ops into the `addmm` that produces them, and folds them into the
reduction that consumes them.

## What it actually does against the lazy version

`fuse.jl` computes, per graph, the set of values that may stay lazy — i.e. the
ones it fuses at run time. That set is directly comparable to the ops this pass
removes, so the comparison needs no estimating:

```
                  lazy defers   this pass removes   still lazy after
sam2_encoder            1          1  (1 fused)            0
sam2_decoder            6         16  (5 fused, 11 epi)     0
MatAnyone (8)          73         51  (47 fused, 4 premap)  6
```

So it captures most of what the lazy version was doing — all of it on SAM 2, 47
of 73 on MatAnyone, with 6 values still deferred at run time — and on top of
that folds 11 epilogues and 4 premaps, which the lazy version has no mechanism
for at all: it can only defer an elementwise op into another elementwise op's
broadcast, never into a GEMM's write-out or a reduction's map step. The 17 ops
of fusion the pass declines on top of that are not conservatism but the shape
guard below: producers *smaller* than the group's output, whose absorption
would recompute them once per broadcast element.

**An earlier version of this document said the opposite** — "the lazy fusion is
already at its ceiling, a rewrite gains two ops across two entire models, so
this is not an optimisation". That rested on a table whose MatAnyone row (1069
ops, 621 elementwise) was measured at a different pipeline stage from its SAM 2
rows, and on a "ceiling" defined differently from what either implementation
does. On one consistent basis the numbers are 711/353 for MatAnyone, and the
conclusion does not survive.

The saving is still modest — 68 ops of 1409 — and it should not be oversold
either. But the honest claim is "captures most of the lazy version and adds two
folding sites it cannot reach", not "gains two ops".

What it buys beyond that is that the graph knows. The lazy version defers
materialisation and never tells anyone, which is why `lifetimes` walks fusion
chains with depth-64
guards to find when an operand is really last read, `planslab` special-cases
values with no storage (worth **1334 MB of a 2020 MB slab** if got wrong), and
discovery has to see through the same laziness. A fused chain that is one op with
declared inputs needs none of that: Mantle's `Liveness`/`Place` see one pass with
one interval, like any other.

It also makes the kernel set enumerable. A nested `Broadcasted`'s type encodes the
whole tree and exists only once execution has nested it, so it cannot be listed
ahead of time — which is why the frozen kernel cache must be filled by running a
workload. A `FusedOp` is built by the pass, at load, from the graph.

## `FusedOp`

```julia
FusedOp((+, *), ((In(1), In(2)), (Tmp(1), In(3))))   # (a+b)*c
```

`In(i)` / `Tmp(i)` / `Konst(v)` operands, so it expresses DAGs rather than chains.
A plain callable, deliberately: it has to be applicable in three places — a
standalone elementwise kernel, a GEMM's write-out epilogue (`matmul!`'s `epi`),
and a reduction's map step.

Three things it needs that were not obvious:

**`@generated` unrolling.** The natural version threads temporaries through a
tuple that grows per function. Type stable at every depth — `@code_warntype` is
clean — and it stops inlining past three, so the tuples heap-allocate: 0 / 0 / 48
/ 112 / 176 / 304 bytes at depths 2..8. Per element inside a kernel that is not
slow, it is broken. Unrolled, every temporary is an SSA value and it is 0 bytes at
every depth, compiling to straight-line `fmul`/`fptrunc`/`fadd`.

**`Cast{T}`, not the type itself.** Putting `Float16` in `funcs` types the slot as
`DataType`, which is not a concrete callable — a dynamic dispatch per element. And
`_to_copy` is the most frequent op in an exported graph.

**`Rounded{T,F}`.** Fusion must round where the unfused chain *stored*. Exported
graphs carry scalars as plain numbers, and a Float64 one promotes: `x + 1.0e-6` on
a Float32 tensor evaluates in Float64, and unfused it is immediately stored into a
Float32 destination. Fused, the Float64 carries onward. On
`(2 - sqrt(x + 1e-6)) * 3` at `x = 4` the subtraction cancels to ~1e-7 and the two
rounding orders differ in the fourth significant digit. Every step gets a
`Rounded`; it is free where it changes nothing.

## The bug worth remembering

The first version built arguments from `op.ins`. But `operand` resolves a
*position* to either a scalar attribute (`arg$(pos-1)`) or the next unconsumed
tensor — so `add.Tensor(x, 1e-6)` has one entry in `ins` and the scalar in attrs.

Dropping it produced `+(x)`. That is not an error: it is unary plus, the identity.
And `sub.Tensor(x, 1)` became `-(x)`, negation. SAM 2's decoder came out with mask
logits off by **1.96** and IoU off by **0.30** — numbers that read exactly like a
precision problem, which is where the next hour would have gone.

`operandrefs` now mirrors `operand`, and `recipe` asserts arity, because the
failure mode is a function that runs and returns the wrong answer.

## Conservative by construction

Left unfused rather than half-modelled: `add`/`sub` with non-unit `alpha`, ops
carrying an `act` from `foldrelu`, symbolic scalars (they need `evalexpr` against
`dims`, unknown when the passes run), and `_to_copy` to an integer type (torch
saturates via `safetrunc`; `T(x)` would throw or differ).

## Epilogue

`foldepilogue` folds a unary elementwise op into the `addmm` producing it. The
GEMM's store already reads and writes every element, so the activation is free
there and a separate op is a full read-modify-write pass. `foldrelu` does this by
name for `relu`/`gelu`; this generalises it to any unary expression.

**`addmm` only**, because it is the one producer whose handler takes `epi` as a
*function*. Convolution's dispatches on `Val(act::Symbol)` inside
`conv_epilogue_kernel!`, so generalising it changes the kernel signature rather
than the caller — and SAM 2 has one conv candidate against eleven for `addmm`.

**Unary only**, because `epi` sees one value inside the store: `mul.Tensor(x, 2)`
folds, `mul.Tensor(x, y)` cannot.

## The shape guard: fusion must not multiply work

`fusegroups` grows a group backwards through operands, and its first version
never asked whether the operand it was absorbing was *smaller* than the group's
output. SAM 2's decoder has a layer norm that arrived already decomposed:

```
add_26   (128,128,64,1)   1048576 elements
mean     = mean.dim(add_26, dim=1, keepdim)  -> (128,128,1,1)
mean_1   = mean.dim(pow_1,  dim=1, keepdim)  -> (128,128,1,1)   the variance
add_28   = (add_26 - mean) / sqrt(mean_1 + eps) * gamma + beta
```

Unfused, `add` and `sqrt` run on the 16384-element `mean_1`; the expansion to
1048576 happens one op later, at the `div`. The backwards walk absorbed them
*past* the expansion point, and then that `sqrt` ran 64 times — once per
element of the normalised axis, whose length is exactly the broadcast factor.

Nothing is reordered: the arithmetic per output element is identical either
way. What the walk destroyed is **sharing**. `recipe` collapses the chain into
one scalar function and `fusedemit!` broadcasts it over the union shape, so
every step is evaluated once per output element; that a step's operands only
vary over (h, w) is information the graph had and the flattened callable does
not. In mapreduce terms, `sqrt(var + eps)` is a postmap on the reduction's
output and belongs on the reduction's shape — done inside the consumer's map
it costs the full output shape. This is also not a `Broadcasted` deficiency:
any elementwise kernel with one output element per iteration recomputes it,
hand-written or not. What avoids it is a different iteration space (a real
layer-norm kernel, one block per row), not a smarter broadcast.

The fix is one line: refuse to absorb a producer whose shape is not the group
output's shape. Compared structurally, not by element count, because the passes
run at `Model` construction where shapes can still be symbolic in `h` and `w` —
equal shapes cannot expand, anything else is declined. The walk then seeds a
fresh group on the declined ops, which is exactly the right structure:

```
group 2  (16384 el)   add -> sqrt                  fused, on the SMALL shape
group 1  (1048576)    sub -> div -> mul -> add     the normalize
```

Measured across both models, counting steps whose operands are all smaller than
the group's output — the genuinely recomputed work:

```
before guard   18 recomputed steps   4,425,642 redundant element-evaluations
after  guard    0                    0
```

The cost is 17 ops of fusion forgone (16 on MatAnyone, 1 on the decoder), which
is where the table at the top loses ground against the lazy version. An A/B
with the guard disabled — the same code that produced every earlier number in
this document — isolates what it changes arithmetically:

- **MatAnyone: nothing.** All 8 graphs bit-identical guard on vs off, and the
  end-to-end delta against unfused is the same with and without it (see the
  tie-flip correction below). Materialising the small tensor and recomputing it
  per element produce the same bytes here.
- **SAM 2's decoder: one ulp-reshuffle, 6.7e-6.** Splitting the group moves an
  FMA-contraction boundary, so a few elements round differently — the same
  class and size as fusion's own 5.7e-6 against unfused, which is unchanged to
  the last digit. `verifygraph`'s decoder row is likewise unchanged:
  `add_8`, 0.018786251544952393, guard on and off.

## Results

All counts on the same basis: a `Model` built with `fuse = false` against one
built with `fuse = true`, both after every other rewrite pass.

```
                    ops        removed   fused  epilogues  premaps
sam2_encoder      549 -> 548       1        1       0         0
sam2_decoder      149 -> 133      16        3      11         0
SAM 2 total       698 -> 681      17                              -2.4%
MatAnyone (8)     711 -> 660      51       32       0         4    -7.2%

runsam2 score  0.878969  — unchanged (SAM2Runner's maskatframe gate, 24 assertions)
DNNKernels suite  9022 assertions green (55 in test_fusepass.jl)
Mantle suite      green
```

The `fused` column counts groups, not removed ops; with the shape guard the
decoder's one layer-norm group becomes two (the small `add -> sqrt` and the
normalize), which is why it reads 3 where it read 2.

An earlier version of this table said "MatAnyone 1069, 65 ops". 1069 was the
count from the elementwise *analysis*, not the op count of the graphs being
compared — two different bases in one table.

## Verifying it: against what shipped, not against materialisation

Per graph, same inputs, `D.call` — the production path, with the lazy fusion and
the slab, which is the configuration the parity gate would test. Inputs are the
reference inputs from the `matanyone-refs` artifact at its manifest resolution
(128x227, so `dims = (h=8, w=15)`), regenerated per run; at this width every
convolution takes `splitk = 1`, so the controls are clean:

```
                       ops        control(old,old)   old vs new
encode_image           61->60     0.0                0.0
encode_mask_deep      101->88     0.0                0.0
encode_mask_shallow    90->83     0.0                0.0
pixel_fusion           27->25     0.0                0.0
pred_uncertainty         7->7     0.0                0.0
readout_query         340->323    0.0                0.0
segment                76->65     0.0                5.9604645e-8
transform_key           9->9      0.0                0.0
```

**Identical on seven, one fp32 ulp on the eighth.** `segment`'s 5.96e-8 is
2^-24 — a single Float32 ulp — from the guard moving an FMA-contraction
boundary, the same mechanism the decoder shows below. An earlier version of
this table (synthetic 8x8 inputs, pre-guard op counts) had two nonzero rows
equal to their own controls; those were the split-K convolution below, and the
table's current basis has no split-K in it.

SAM 2, same method, reference inputs:

```
                       ops        control(old,old)   old vs new
sam2_encoder          549->548    0.0                0.0
sam2_decoder          149->133    0.0                5.722046e-6   (rel 3.8e-7)
```

The encoder is exact. The decoder moves by about **three fp32 ulps** on an
output scaled to 14.9 — the same number to the last digit as before the shape
guard, because the element that sets it sits in the `mul -> add` tail both
groupings share — and `runsam2`'s score is unchanged at 0.878969.

It is not the epilogues, which was the obvious suspect at 11 of them. Running
`fuseops` alone, with `foldepilogue` skipped, gives `5.722046e-6` — the same
number to the last digit (op counts below are the pre-guard ones; the guard
moves each total up by one, as the table shows):

```
unfused vs fuseops only            5.722046e-6     149 -> 143
unfused vs fuseops + foldepilogue  5.722046e-6     149 -> 132
```

So folding 11 unary ops into their GEMMs' write-out costs nothing measurable,
and the difference is entirely `fuseops`.

Bisecting the decoder's two groups, one op at a time against the unfused
baseline over all 448 buffers, names the one responsible (measured before the
shape guard; the guard splits group 1 at the `sqrt`, into the small statistics
group and the normalize, and the tail that causes this is in the normalize):

```
group 1   7.6293945e-6   add -> sqrt -> sub -> div -> mul -> add
group 2   0.0            mul -> sub
```

Group 1 is the layer-norm shape, and it ends `t5 = x4 * t4; t6 = t5 + x5`.
Replacing those last two steps with a single explicit `fma` and broadcasting
both on the device gives **bit-identical output** — so the contraction is
confirmed here rather than inferred from the magnitude:

```
fo1 (mul then add) vs explicit-fma variant   0.0
unfused vs either                            4.7683716e-7
```

Reconstructing the chain on the CPU is no use for this: the device's `sqrt` and
`/` differ from the CPU's by the same 4.77e-7, so all three CPU variants sit
equidistant and discriminate nothing. The GPU-against-GPU comparison is the one
that answers it.

Two ways of measuring this gave two wrong answers first, and both are easy to
repeat:

**Reusing one set of input arrays across the runs being compared.** Some graphs
write into their inputs, so run 2 sees what run 1 left. That reported `0.0` for
three graphs that did differ. Inputs get regenerated per run now.

**Comparing against `lazy = Set{String}()`.** Forcing every intermediate to
materialise looks like the neutral reference and is not: it is a configuration
that never runs. Against it, three graphs differ by ~2.7e-3, and none of that is
this pass's doing.

## Why fusion cannot be bit-equal to materialised execution

`sigmoid -> mul -> add`, all Float16, in `encode_mask_shallow`. The fused result
was one Float16 ulp (2^-10) off the materialised one. It is neither a wrong
recipe nor a lost `Rounded`:

```
GPU unfused            vs per-step-rounded CPU : 0.0
executed FusedOp on CPU vs per-step-rounded CPU : 0.0
GPU fused              vs per-step-rounded CPU : 0.0009765625
GPU fused              vs fma(x2, t1, x3)      : 0.0        (16384 of 16384)
```

The device contracts the `mul` and the `add` into an **FMA**, straight through
the `Float16` conversion between them. One rounding where materialising does
two — *more* accurate, and not something a `Rounded` can prevent, because the
contraction happens in the SPIR-V backend and not in the Julia code.

Isolating it took narrowing the chain: `sigmoid` alone and `sigmoid -> mul` are
both exact on device, and the third step is where it appears. Singleton
broadcast dims and 4-D shapes were both ruled out first; the flattened 1-D case
differs identically.

**So the lazy fusion has always done this too** — it nests into one kernel and
contracts the same way — which is why the two agree exactly in the table above,
and why a parity gate that passed before still means something. Bit-equality
with materialised execution was the wrong target: any real fusion of an fp16
chain forfeits it by construction.

## The PyTorch gate

### MatAnyone — it runs, and it passes

I twice recorded this as unrunnable, on the grounds that `matanyone-refs` was
"bound in no `Artifacts.toml`". **It is bound**, in
`MatAnyoneRunner/Artifacts.toml`, with a download URL to the `assets-v1`
release. What had happened is that it was bound but not *installed*, and the
download failed once — which is indistinguishable from an unbound artifact at
the call site, because `matanyoneprecisions()` returns empty for both. The
second attempt succeeded immediately.

Two stale comments pointed the same way and are now corrected: `refsdir`'s
docstring said "Not bound yet", and `DNNKernels/test/runtests.jl` asserted the
artifact "is bound in no Artifacts.toml in this repository".

Layer by layer against PyTorch, all 8 graphs, both precisions, with the fusion
passes applied and without:

```
autocast   UNFUSED  all 8 pass      FUSED  all 8 pass
fp32       UNFUSED  all 8 pass      FUSED  all 8 pass
```

"FUSED" here is the three fusion passes applied to the **raw exported graphs**
— the only basis on which per-node comparison means anything, since the
references are keyed by the raw node names. The full `Model` pipeline cannot be
checked this way: `foldbatchnorm` folds the BN into the conv, so the op still
named `convolution` then computes conv+BN against a reference that holds the
conv alone (first mismatch max|Δ| 22.3, unfused included — a keying artefact,
not a kernel bug). The production pipeline's fused-vs-unfused equivalence is
the per-graph table above; this gate is the one that anchors both to PyTorch.

`MatAnyoneRunner/test/test_parity.jl` now records **61 assertions** where it
used to record 1 and stay green. The fp32 error floor comes out where its
docstring says it should — 3.1e-6 on `encode_image`, 5.9e-5 on
`encode_mask_deep` — and `segment`'s worst is 0.0032 at `convolution_21`, which
is the split-K convolution from the section below, showing up independently in
a test written long before anyone looked at it.

This is the gate this whole pass needed, and it is green with fusion on.

### SAM 2

SAM 2's refs are bound too (1515 entries), so `verifygraph` runs there as well,
fused against unfused:

```
                 ops        result                first mismatch
decoder fused    133        1 mismatch            add_8    max|Δ| 0.018786251544952393
decoder unfused  149        1 mismatch            add_8    max|Δ| 0.018786251544952393
encoder fused    548        1 mismatch            add_129  max|Δ| 0.36749267578125
encoder unfused  549        1 mismatch            add_129  max|Δ| 0.36749267578125
```

**Identical to the last digit.** Both mismatches predate this pass; fusion moves
neither.

One cautionary measurement: a single `verifygraph` run in an earlier session
reported `add_129` at max|Δ| **4.72** — 13x this value — on the fused encoder
right after the shape guard landed. It never reproduced: three reruns in that
session and five more in a fresh one, across fused and unfused graphs and
crossed weight pairings, all read 0.36749267578125 to the last digit, and the
encoder's graph and outputs are bit-identical with and without the guard. That
session had five models resident on a shared-memory APU under GC pressure; the
reading is recorded here as environment noise, not waved away — if 4.72 ever
shows up again, it is not this pass.

Worth having diagnosed, since a gate that fails is a gate nobody reads. `add_8`
sums `permute_1` and `permute_2`; the first matches the reference exactly and
the second is off by 0.018786252, which passes straight through the addition
unchanged. Following `view_9 -> view_7 -> clone_1` ends at a **weight** — a
constant-folded tensor, Float32 for us and Float16 in the reference dump.
Rounding ours to Float16 does not reproduce theirs, so it is not quantisation of
our value: torch folded that constant *in* fp16 and we fold it in fp32.

Ours is the more accurate one. It is flagged because `verifygraph` picks its
tolerance from the checked tensor's dtype — `view_9` itself passes under
`rtol16 = 3e-2`, and then `add_8`, which is Float32 on both sides, is held to
`rtol = 1e-3` while carrying an error of fp16 size. An fp32 op fed by an fp16
constant fails a gate that is right about every tensor it looks at.

The encoder's `add_129` cannot be localised at all, and finding out why is the
more useful result:

```
sam2_decoder   149 ops, 133 with a reference node   89.3%
sam2_encoder   549 ops,   3 with a reference node    0.5%
```

All 543 ops before `add_129` have no reference node, so its 0.367 is the
accumulated result of 543 unchecked ops — mostly long fp16 `add` chains and
fp16 convolutions feeding the neck. Tracing its operands back eight levels
reaches `add_114` without meeting a single reffed node.

`verifygraph` prints `[543/548] FIRST MISMATCH`, which reads like 543 nodes
passed. Two did. That ordinal is a position in `g.ops`, not a count of
comparisons — the second time in this session a progress-looking number turned
out to be measuring something else.

**The sparse coverage is deliberate, not missing.** `tools/dump_sam2_refs.py`
says so plainly:

> The encoder is the expensive half — 1491 ops at 1024x1024, whose
> intermediates run to tens of GB — so by default only its six outputs are
> recorded, and the decoder is recorded node by node. `--nodes all` overrides
> that when the encoder is suspect.

So the encoder is checked *at its outputs*, by design, and `add_129` is one of
them. That makes the 0.367 an output-level disagreement rather than an
unverified intermediate — a slightly worse reading than "no coverage", though
`runsam2`'s score is unaffected and it is identical fused and unfused.

Localising it is a supported operation, not a missing capability:

```
uv run tools/dump_sam2_refs.py --nodes all --size <smaller>
```

which is what that flag exists for. It needs a torch install and a SAM 2
checkpoint that are not currently present in this tree — `dev/JuliaVision` has
no `.venv` and torch resolves from a CUDA 12.8 index on an AMD card — so it is
a deliberate ask rather than something to run unprompted. (An earlier version of
this section named `tools/dump_refs.py`; that one is MatAnyone's.)

**Neither is caught by CI.** `SAM2Runner/test/runtests.jl` reads the refs but
never calls `verifygraph`, so the layer-by-layer gate is not wired into any
suite — which is how two mismatches stay resident. Wiring it in means deciding
what to do about the two above first, since it would land red.

## Fused graphs through baked Mantle plans

The other thing untested until now, because the Mantle port and this pass were
built alongside each other and `fused.elementwise` is a new op in the captured
sequence. `SAM2Runner.defaultmodel()` builds and bakes in 9.07 s, and:

```
run 1 (builds + bakes)   score 0.45917758
run 2 (replay)           score 0.45917758   mask bit-identical
```

A synthetic image and a centre click, so the score is not comparable to the
0.878969 in Results above — that one is the real test frame. What this shows is
determinism across the bake, not quality: a fused op captures and replays like
any other, and the precompile workload builds its plans from fused graphs.

## An unrelated finding: a nondeterministic convolution

`segment` does not reproduce itself. Two runs on identical inputs differ, and
walking the ops in order finds the first divergence at

```
convolution_21   convolution.default   run-to-run diff = 3.0517578e-5
```

with every op before it bit-identical — including `view_50`, which is a view
buffer and so was invisible to a scan that only compared op outputs. Same
inputs, different output. Nine ops differ in total; the other eight are all
downstream, which is why the graph's outputs move by ~5e-7 between runs.
`encode_mask_deep` does the same at 3.4e-7.

**It is `splitk`.** The kernel reproduces it in isolation, outside the graph, at
the same 3.05e-5 — so it is not a missing barrier. Pre-filling the destination
differently changes nothing and two runs at the *same* prefill still differ, so
it is not an uninitialised read either. `conv_coopmat_plan` declines (fp32), and
`convolution_igemm!` splits the reduction:

```
Cout=768  NPQ=64  CRS=4608  cores=40  ->  splitk = 4     9% of elements differ
same convolution at NPQ=4096          ->  splitk = 1     bit-reproducible
```

Split-K accumulates with `Atomix.@atomic +=`, the order the atomics land in is
not fixed, and float addition is not associative. It is what buys the 4-8x on a
shape with only 64 output pixels, so it is a design point and not a defect — but
it is the floor for anything measured on a graph containing one, and the first
thing to rule out when a change appears to move a small-NPQ result by ~1e-7.
Noted in `conv_implicit.jl` beside the split, where someone measuring will hit
it.

**MatAnyone is where this pass earns its keep, not SAM 2.** 51 ops against SAM
2's 17, and the reason is what the graphs are made of:

```
                       elementwise share of ops
MatAnyone (8 graphs)     353/711   49.6%
sam2_decoder              55/149   36.9%
sam2_encoder             106/549   19.3%
```

Anyone judging the pass on SAM 2's encoder alone would conclude it does nothing.

### The gap to the lazy version is a policy, not conservatism

I had this wrong and repeated it several times: that the lazy version reaches
~75 where this reaches 65 because `alpha != 1`, folded `act`s and symbolic
scalars are refused outright. Counting the elementwise ops those guards refuse
*that would actually have joined a group*:

```
none
```

**Zero.** Not one of those guards costs an op on either model. Every
`fusable -> fusable` edge this pass declines, by reason:

```
producer has >2 readers   107      the lazy version RECOMPUTES these
producer has 2 readers     30      likewise
dtype differs (f16 -> f32)  3
```

So the difference is almost entirely that `fuse.jl` lets a value with several
consumers stay lazy and recomputes it per reader, while this pass requires a
single reader and materialises once. That is a real choice — and for the 107
edges whose producer has *more than two* readers, recomputing means evaluating
the expression three or more times to avoid one store.

It is not a shortfall to be closed. Calling it conservatism was a guess that
survived three retellings because it sounded like the kind of thing that would
be true.

The remaining 3 are the dtype guard, which is the one guard that does earn its
place: widening it moved SAM 2's IoU from 1.00000/0.99972/0.97727 to
0.98750/0.99978/0.95556.

Pipeline order in `driver.jl` is `fuseops` then `foldepilogue`, **last**. Fusing
before `hoistcasts` would fuse casts about to be deleted; folding epilogues before
`fuseops` would absorb only a chain's last link.

## Premap

`foldpremap`, the mirror of `foldepilogue`: that one folds a unary elementwise
op into the op that *produces* its input, this one into the op that *consumes*
its output. `sum(f.(x))` becomes `mapreduce(f, +, x)`, and the reduction reads
the operand where it used to read the mapped result.

No new kernel. `sum(f, a; dims)` *is* `mapreduce(f, +, a; dims)` and
`prod(f, a; dims)` is the same with `*`, so the handlers pass the `FusedOp` one
level down, behind a barrier for the same reason `addmm_epi!` has one — the
callable comes out of a `Dict{String,Any}`, and without the barrier the
per-element call inside the reduction would be a dynamic dispatch. `any`/`all`
are excluded: premapping into them changes the question, not the method.

### What it finds, and why that is the interesting part

Across all ten graphs of both models, every reduction:

```
producer has >1 reader              111
producer not unary-fusable           16
no producer (a graph input)           5
candidate                             4
```

**114 of those producers are `add.Tensor`** — the residual stream. In
`x = x + sublayer(x)` followed by `norm(x)`, the sum feeds both the norm and the
next skip connection, so it has to exist as a tensor no matter what. That is a
property of transformers, not of this pass, and no widening of the predicate
reaches it. Worth stating as a structural result rather than a disappointing
count.

The four that do fire are all `prod(1 - x)`, three in MatAnyone's
`readout_query` and one in `segment`.

### The earlier "0 candidates" was wrong twice

It counted only `sum`/`mean` and only producers that were literal unary aten
ops — but after `fuseops` a collapsed chain is a `fused.elementwise` op, which
that predicate does not match, and `prod` was never in the set at all. `prod` is
premappable with no kernel change, and it is where all four candidates live.

The second error was worse: having found that `native_layer_norm` and `_softmax`
are the ops consuming elementwise results in bulk and that folding into *those*
needs a kernel-signature change, I let that stand in for premap as a whole and
shipped nothing. One hard sub-case is not a reason to skip the easy one.

### And the hard sub-case turns out to be empty

Having said that, the honest version of the same claim needed checking too.
Per reduction:

```
                            total   single-reader src   premappable
native_layer_norm.default     114          12                0
mean.dim                       11           1                0
sum.dim_IntList                 4           0                0
prod.dim_int                    4           4                4
_softmax.default                3           3                0
```

All 15 of those single-reader producers are **binary** — 12 `add.Tensor` with
two tensor operands, 3 `bmm.default`. A map step is applied per element to one
value, so a two-tensor add cannot be one no matter what the kernel signature
accepts.

So teaching `native_layer_norm` and `_softmax` to take an `f` would fold **zero**
ops on both models. That is now the reason they are excluded, and it is a much
better one than "the kernels are hand-written": it is a fact about the graphs
rather than about the effort.

### Verification

Production config, control column, fresh inputs per run (op counts pre-guard;
the shape guard moves the fused totals — `readout_query` fuses to 323 now, not
320 — the numerics are the point and are unchanged):

```
                   ops        control   no-premap vs premap
readout_query     320->317    0.0       0.0        EXACT, 3 premaps
segment            66->65     2.68e-7   2.68e-7    within own noise, 1 premap
(six others)      unchanged   —         —          no premap
```

`encode_mask_deep` is the useful accident here: it has *identical graphs* in
both models and still differs by 2.4e-7, which is the split-K convolution
proving itself again.

End to end, `runmatanyone` at 128x128 — seed, two warmup frames, one ordinary
propagation, so `readout_query` and `segment` run inside the stateful loop with
a real memory bank rather than one graph at a time:

```
control (no-fuse vs no-fuse)   0.0
no-fuse vs fused + premap      0.0
```

Bit-identical. At this resolution the convolutions take `splitk = 1`, which is
why the control is clean here and not on the 8x8 per-graph runs.

**That bit-identity does not survive a different input draw, and the shape-guard
A/B is what exposed it.** Re-run with a fresh random frame and mask, fused vs
unfused differs: max 0.0125, on 407 of 16384 matte elements — with the guard
*and* without it (0.0126, 408 elements), so the guard is not the cause. The
mechanism: per-graph, fused vs unfused differs by at most one fp32 ulp
(`segment`, above); `readout_query` has tie-sensitive predicates — the parity
gate's own log shows `ge_13` flipping 1 of 120 elements against torch — and one
flipped selection reroutes a memory-bank row read, which the stateful loop then
amplifies. Whether a given input draw contains a near-tie sitting within one
ulp is luck; this draw did, the earlier one did not. The honest end-to-end
statement is: deterministic (control 0.0), correct per-graph to one ulp, and
bit-identity between fused and unfused holds exactly when no tie flips.

`unaryfused` now wraps in `Rounded`, because folding removes the store that
would have rounded to the declared dtype, and a reduction consumes the value
with no store in between. For an epilogue it changes nothing — the GEMM's own
store rounds immediately after — and SAM 2's decoder confirms it, still
`5.722046e-6` against the unfused baseline, unchanged to the last digit.

Pipeline order is `fuseops` → `foldepilogue` → `foldpremap`. Epilogue before
premap because an op sitting between an `addmm` and a reduction can be folded
either way but only once, and the GEMM's store touches those elements whether or
not anything rides along, while a reduction's map step is work either way.
