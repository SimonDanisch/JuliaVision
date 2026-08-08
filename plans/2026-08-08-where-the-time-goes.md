# Where the time actually goes

Measured after the fusion pass and the Mantle port landed, because "it all runs
and the tests are green" is not an answer to "is it fast".

Method: warmup runs, `synchronize` inside the timed region, 15–25 reps, full
distribution rather than a mean. Card is an AMD Radeon 8060S (RADV STRIX_HALO),
an APU sharing system memory.

## Fusion does not make anything faster

```
MatAnyone runmatanyone  @128x128    no-fusion 48.8   fused 49.2   1.00x
MatAnyone runmatanyone  @512x512    no-fusion 317.4  fused 317.5  1.00x
SAM 2 encoder           @1024       no-fusion 280.0  fused 281.3  0.99x
SAM 2 decoder           @1024       no-fusion 13.7   fused 14.0   0.98x
```

(minima, which are the least noisy estimator here.)

**Zero, to within noise, and marginally negative on three of four.** That is the
result the fusion spec predicts — it removes 85 ops of 1409, and op count is not
what this is bound by — but it had never actually been timed, only argued.
Anyone tempted to sell the pass on speed should read this table first. Its case
is that the graph knows what was fused, which is a correctness and planning
property, not a throughput one.

## Baking cuts the tail, not the floor

SAM 2 encoder at 1024x1024, same graph, same fusion state, eager per-op dispatch
against a baked Mantle plan:

```
              p0     p25    p50    p75    p100
baked      268.8  272.2  274.6  276.6  283.0     5% spread
eager      282.0  288.3  294.2  409.9  432.0    53% spread
```

The floor moves 5% (282 -> 269). The **p75 moves 1.5x** and the worst case 1.5x,
because the eager path's per-op CPU dispatch produces a long tail that the baked
one simply does not have. For a frame-time budget that is the number that
matters: 283 ms worst case against 432.

So the honest claim for baking is predictability. It does not make the GPU do
less work — it is the same kernels in the same order — it stops the CPU from
being in the loop for each of them.

## The encoder's profile is the right shape

Per-op device time, serialising measurement (`diag.optimes`), SAM 2 encoder:

```
addmm.default                       195 calls   120.9 ms   38.8%
_scaled_dot_product_flash_attention   48        120.5      38.7%
add.Tensor                            98         18.6       6.0%
clone.default                         90         17.3       5.5%
native_layer_norm.default             96         14.7       4.7%
convolution.default                    7          8.8       2.8%
max_pool2d_with_indices                6          7.9       2.5%
```

77.5% in GEMM and attention is what a transformer encoder should look like, and
all 195 `addmm` are `Float16 x Float16 -> Float16` on a device reporting
`coopmat = true`, so they are on the tensor-core path rather than a scalar
fallback. Nothing here is pathological.

Two things are visible as overhead rather than work:

**`clone.default`, 90 calls for 17.3 ms** — pure copies, costing more than every
convolution and pooling op combined. Copy elision is worth a look.

**`add.Tensor`, 98 calls for 18.6 ms** — the residual streams. These are exactly
the ops `fuseops` cannot touch, because each feeds both a norm and the next skip
connection (see the fusion spec's premap section). 6% of the encoder sitting in
the one shape the fusion pass structurally cannot reach is worth knowing.

## Resolved: 270 ms against an in-repo note of 118 ms — different hardware

`Ctx`'s `clampattn` docstring records, for this same call on this same class of
card, `encode 118.63 -> 120.75` ms. The measurement above is **269 ms** for the
baked path and 281 eager — 2.3x slower than that note.

Ruled out before asking:

* **Not fusion** — the A/B above shows fused and unfused within 1%.
* **Not the Mantle port** — baking is *faster* than the eager path, not slower.
* **Not memory pressure** from the seven models left resident in the measuring
  session — a full `GC.gc(true)` moves it from 268.8 to 269.5, i.e. not at all.
* **Not a scalar GEMM fallback** — all 195 are fp16 on a coopmat-capable device.

The answer was the first "not ruled out" candidate: the note was measured on an
**Ada 4000**, not on this APU. 2.3x is a plausible high-end-discrete to
integrated-APU ratio for a transformer encoder, so there is no regression to
find here — but the docstring should say which card its numbers came from, and
now does.
