# benchmark/

Every ported model's forward pass, this tree against PyTorch, on one card,
tracked over time.

```
julia --project=../.. benchmark/suite.jl                  # everything, both runtimes
julia --project=../.. benchmark/suite.jl sam2-encode rife # a subset
julia --project=../.. benchmark/suite.jl --no-torch       # this tree only
julia --project=../.. benchmark/suite.jl --compare results/latest.json
```

`--project` is the **repo root**, not `JuliaVision`: the runners and
`BenchmarkTools` are both resolvable there, and adding BenchmarkTools to
JuliaVision's own Project would be a dependency added for a benchmark.

## The files

| file | what it is |
|---|---|
| `models.toml` | every model's parameters, written down **once** |
| `julia_models.jl` | builds each model on this backend, asserting the spec |
| `torch_models.py` | builds the same model in PyTorch, asserting the spec |
| `torch.sh` | runs Python under the interpreter that can see this card |
| `suite.jl` | runs both sides, saves results, prints and judges |
| `results/` | one `.json` + `.md` per run, plus `latest.json` |

## Why the parameters live in one file

The harness this replaces declared its shapes twice, once per language, and they
drifted: the PyTorch Kokoro row ran 30 tokens against the Julia row's 50 for
long enough to be committed, and the ratio between them meant nothing while it
did. A number that differs between the two sides is not a slow model, it is a
different program.

So `models.toml` is authoritative and **both sides assert against it rather than
deriving it**. `rife` checks the installed export's padded frame size, `sam2`
checks the checkpoint's image size, `kokoro` checks that its phonemes tokenise
to the stated count. A model that disagrees with the spec is an error, not a
row — which is the opposite of a harness that reports whatever it happened to
build.

## Eager and compiled

Both are recorded, because they answer different questions.

* **eager** is the like-for-like comparison. The graph Lava runs came out of
  `torch.export`, which is eager PyTorch's own decomposition — same
  decomposition, different runtime.
* **compiled** is `torch.compile`, Inductor fusing that graph into kernels this
  tree does not generate. Not a like-for-like anything: it is the target.

A table with only one of them either flatters us or reads as a fantasy gap.
`matanyone-step` records compiled as unsupported with the reason rather than a
number, because `InferenceCore.step` mutates a memory bank across calls.

## Measurement

This card idles at 210 MHz of 3105 and takes **seconds of sustained load** to
boost, so a benchmark that starts measuring before it does reports the ramp.
Measured at a 2 s ramp the SAM 2 row read 69.4 ms against a 14 s ramp's 67.3 on
the same card minutes apart, and a longer ramp giving the *faster* number is the
signature of a card still climbing. Both sides ramp for `suite.ramp_seconds`
before recording anything.

The Julia side additionally gates each sample on the clock: `tools/measure.jl`
brackets every sample with a clock reading and discards any that dipped below
`suite.clock_floor` of the measured plateau. A row whose samples were all
rejected prints with `!` — it is not broken, it is host-bound (`neurallut` is
6 ms of mostly host work, so the card idles inside its own sample), and the
number is worth less than a gated one and more than nothing.

## Results

`suite.jl` writes `results/<stamp>-<host>.json` (a `BenchmarkTools`
`BenchmarkGroup`) and `results/<stamp>-<host>.md` (the table plus the conditions
it ran under), and copies the JSON to `results/latest.json`.

The JSON is BenchmarkTools' own format, so regression checking needs no bespoke
comparator:

```julia
using BenchmarkTools
old, new = load("results/a.json")[1], load("results/b.json")[1]
judge(median(new), median(old))
```

The leaves are `"julia"`, `"torch-eager"` and `"torch-compiled"` under each
model, so `judge` also compares the two runtimes against each other within one
run. The timings are real per-sample times, not BenchmarkTools' own
`@benchmark` — that tunes an evaluation count against a fast repeatable
function, and these are GPU forwards where the clock has to be held up and
individual samples thrown away when it is not.

## The PyTorch interpreter

The project's own `.venv` is a **CUDA** wheel (`+cu128`) and
`torch.cuda.is_available()` is `False` here, so it can export graphs and cannot
time anything. `torch.sh` runs a ROCm interpreter instead:

```
JULIAVISION_TORCH_PYTHON    default ../tmp/baseline/.venv/bin/python
JULIAVISION_TORCH_PRELOAD   default the system ROCm HSA + HIP pair
```

The preload is not optional: the ROCm wheel bundles its own runtime, that one
segfaults at the first kernel launch on this gfx1151 box, and the system ROCm
7.1 launches the same kernels fine (verified with a plain hipcc program).

**The default interpreter lives under `tmp/`, which is scratch and is not backed
up.** If it is gone:

```bash
python3 -m venv tmp/baseline/.venv
tmp/baseline/.venv/bin/pip install --index-url https://download.pytorch.org/whl/rocm7.1 torch
# plus whatever a given model's setup imports: transformers, kokoro, hydra-core, …
```

or point `JULIAVISION_TORCH_PYTHON` at an interpreter that already has a
GPU-visible torch. `suite.jl` reports the failure and prints our own times
without a denominator rather than pretending the comparison happened.

### When the PyTorch side reports a reason instead of a number

That is the suite working. Every torch cell that is empty has a line under the
table saying why, and on this machine as of 2026-09-22 all seven do. They are
two different problems and only one of them is about Python packages.

**Missing upstream checkouts and weights.** `depthanything`, `neurallut`,
`matanyone-step` and `rife` need `dev/Depth-Anything`,
`dev/Image-Adaptive-3DLUT`, `dev/matanyone2` and RIFE's `flownet.pkl`
respectively, none of which are on this box. `sam2-encode` has its checkout but
the `sam2` package is not importable from the benchmark interpreter. These are
`git clone` and weight downloads; `tools/models.py fetch` covers some of them and
`plans/models-to-port.md` records the one it does not.

**Compiled extensions cannot be layered.** `whisper-encode` and `kokoro-speak`
fail with `Could not import module 'WhisperForConditionalGeneration'`, which is
transformers masking the real error: `operator torchvision::nms does not exist`.
The project `.venv`'s torchvision is a compiled extension built against its CUDA
torch, and its operator registry will not load under the ROCm torch. Pure-Python
packages layer fine; this one has to be installed into the ROCm interpreter
against the torch that is actually running there:

```bash
tmp/baseline/.venv/bin/pip install --no-deps \
    --index-url https://download.pytorch.org/whl/rocm7.1 torchvision
```

`--no-deps` on purpose: without it pip is free to move `torch`, and that
interpreter's torch is the one thing about it that is known to work here.

## Adding a model

1. Add a `[models.<name>]` block to `models.toml` with its parameters.
2. Add a builder to `julia_models.jl` — it will not load without one, by design.
3. Add a setup to `torch_models.py` — likewise.

Both sides fail loudly on a model the spec declares and they do not implement,
so the spec cannot get ahead of the code.
