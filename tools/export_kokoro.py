"""Export Kokoro-82M (text -> speech) to DNNKernels' graph JSON.

    uv run tools/export_kokoro.py --tokens 96 --frames 640

## Two graphs, because the output length is computed inside the model

`KModel.forward_with_tokens` does this in the middle:

    duration = torch.sigmoid(duration).sum(axis=-1) / speed
    pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
    indices  = torch.repeat_interleave(torch.arange(n_tokens), pred_dur)
    pred_aln_trg = torch.zeros((n_tokens, indices.shape[0]))

`indices.shape[0]` is `sum(pred_dur)` — **how long the audio is, decided by the
model from its own logits**. `torch.export` cannot bake that, the same way it
could not bake Whisper's decode loop. So the model is cut there:

  * **`kokorotext`** — `(input_ids, ref_s) -> (d, t_en, duration)`. Everything
    up to and including the duration logits. Static in the token count.
  * host — round the durations, expand them into a per-frame token index.
  * **`kokorovoc`** — `(en, asr, s) -> audio`. The prosody predictor's
    frame-rate half and the whole iSTFTNet vocoder. Static in the frame count.

**The alignment is a gather, not a matmul.** `pred_aln_trg` is one-hot, so
`d.T @ pred_aln_trg` just repeats token `i`'s column `pred_dur[i]` times. The
reference materialises an `n_tokens x n_frames` matrix and multiplies; the host
here builds the index vector and gathers. Same answer, no matrix.

## Why the token count is fixed per export, and padding is not an option

Both `TextEncoder.forward` and `DurationEncoder.forward` call
`nn.utils.rnn.pack_padded_sequence`, which does not export — it produces a
`PackedSequence` whose extent depends on the data. At batch 1 with no padding it
is an identity, so the wrappers below call the LSTMs directly instead. That is
the same move the Whisper decoder export needed: write the step out against the
model's own submodules rather than fight a wrapper built for training.

That leaves the token count fixed at export time, and **padding cannot
generalise it**: all six of Kokoro's LSTMs are *bidirectional*, so a reverse pass
over a padded tail starts inside the padding and contaminates every earlier
position. Length generalisation therefore needs one export per bucket (or a
hand-unrolled length-aware LSTM, which at 512 timesteps x 6 LSTMs x 2 directions
is tens of thousands of ops). **This exporter does one length; bucketing is the
next step and is not done here.**

## What is left on the host

Phonemisation. `KPipeline` uses `misaki`/espeak to turn text into the phoneme
string that `KModel.vocab` maps to ids — a lookup table and a G2P, no tensors.
"""

import argparse
import contextlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

import export_graphs as EG

from common import find_root
ROOT = find_root()
GEN = ROOT / "gen"
WEIGHTS = GEN / "kokoro"


def lstm(mod, x):
    """`mod(x)` with the packing stripped.

    `pack_padded_sequence` at batch 1 and full length reorders nothing and drops
    nothing, so the packed call and this are the same function. It is written out
    because the packed one does not trace.
    """
    y, _ = mod(x)
    return y


def exact():
    """A scope that stays in fp32 even under autocast.

    Two blocks of this model need it, and which two was measured rather than
    guessed — see `tools/precision_blame.py`. Rounding one submodule at a time to
    fp16 and comparing the audio:

        decoder.decode      27.9 Melem   1.000009   <- every convolution
        decoder.generator   19.7         0.998773
        text_encoder         5.6         1.000009
        predictor.lstm       1.8         1.000010   <- the LSTMs are NOT sensitive
        predictor.N          3.3         1.000010
        bert                 6.3         0.962368   <- costs
        predictor.F0         3.3         0.925794   <- costs

    The arithmetic is free; the damage is in `bert` and `predictor.F0` alone.

    **`F0` is the one with a mechanism, and it is why the loss grows with the
    utterance.** The vocoder's excitation is `sin` of a *cumulative* F0, so an
    error there accumulates PHASE along the sequence. Whole-model fp16 falls
    0.9901 -> 0.8662 -> 0.7549 across medium, long and very long inputs, while
    every policy holding F0 in fp32 stays flat at 0.994+.

    Keeping these two exact leaves **88% of the weights in fp16** — including 81
    of the 90 convolutions, which is where the time goes.

    **Not all 90, and this used to claim otherwise.** `predictor.F0` is held
    exact, and its blocks CONTAIN convolutions — the AdainResBlk1d stack plus
    `F0_proj` — so nine convolutions ship with fp32 weights:

        3x (256, 256, 3)   2x (512, 512, 3)   1x (512, 1, 3)
        1x (256, 512, 3)   1x (256, 512, 1)   1x (1, 256, 1)  <- F0_proj

    That matters on the Lava side: `conv_coopmat_plan` gates on both operands
    being fp16, and the target device has no fp32 tensor-core path at all, so
    those nine decline the cooperative-matrix kernel by construction.

    Whether they NEED to be exact is untested. The mechanism above is
    accumulation of phase in `cumsum`/`sin`; `precision_blame.py` rounds a whole
    submodule at a time, so it cannot separate "F0's convolutions need fp32" from
    "F0's ACCUMULATION needs fp32". If the convolutions can go fp16 while the
    cumsum/sin stay exact, nine of them move onto the tensor cores — but that
    must be validated with the same long-utterance cosine check this policy was
    chosen by, not assumed.

    bf16 was tried first and is WORSE, not better: 0.4462 against fp16's 0.8662
    on a long utterance. Its fp32 exponent range does not help because these
    models are precision-limited, not range-limited — 7 mantissa bits against 10.
    """
    return torch.amp.autocast(device_type="cuda", enabled=False)


def keepexact(obj, *names):
    """Force `obj`'s named methods to run in fp32 even under autocast.

    For the blocks that are reached from inside another module's `forward` and so
    cannot be wrapped at the call site — here, the generator's STFT.

    **The iSTFT has to stay fp32 for a reason beyond accuracy: it goes COMPLEX.**
    `TorchSTFT.inverse` computes `spec * exp(1j * phase)`, and under autocast that
    is `complex32`, which `DNNKernels` has no dtype for — the export loads with
    `KeyError: key "complex32" not found` and names a buffer rather than the op or
    the reason.

    Adding `complex32` would be the wrong fix twice over. An FFT is not a
    cooperative-matrix operation, so half precision buys it nothing, and Lava's
    FFT kernels are `ComplexF32` throughout; the overlap-add would lose precision
    for no speed at all.

    **The arguments are cast up, and that is the whole point of this over a bare
    `exact()`.** `autocast(enabled=False)` stops *new* ops being run in half
    precision; it does **not** restore tensors that arrived already fp16. Here
    `magnitude` and `phase` are produced by autocast convolutions upstream, so
    `magnitude * exp(phase * 1j)` is `complex32` no matter what scope it runs in
    — wrapping the call in `exact()` alone changed nothing at all, and the
    re-export came back with the same two complex32 buffers.
    """
    for name in names:
        fn = getattr(obj, name)

        def wrapped(*a, _fn=fn, **kw):
            up = lambda x: (x.float() if torch.is_tensor(x) and x.dtype == torch.float16
                            else x)
            with exact():
                return _fn(*[up(x) for x in a], **{k: up(v) for k, v in kw.items()})

        setattr(obj, name, wrapped)


def decomptable():
    """Everything decomposed EXCEPT the LSTM.

    `run_decompositions()` unrolls a recurrent layer into one copy of the cell
    per timestep: measured, a 12-step bidirectional LSTM goes from **1 node to
    532**, and Kokoro's six LSTMs are why its two graphs are 7298 ops for a
    30-token utterance. Worse, the count grows with the sequence, so a 256-token
    sentence would be tens of thousands of ops — and at Lava's ~63 us per
    dispatch that is seconds of host time before any arithmetic.

    Keeping `aten.lstm.input` whole makes it ONE op that DNNKernels implements
    with a loop, which is also the only way a bucketed export stays a sane size.
    Everything else still decomposes, so the rest of the graph is the same set of
    primitives every other model here uses.
    """
    tbl = torch.export.default_decompositions()
    for k in [k for k in list(tbl) if "lstm" in str(k)]:
        del tbl[k]
    return tbl


class TextHalf(torch.nn.Module):
    """`(input_ids, ref_s) -> (d, t_en, duration)`.

    Holds the whole `KModel` so weights are named the way its `state_dict` names
    them — the same reason `export_whisper_decoder.py` holds the whole
    `WhisperForConditionalGeneration`.

    `duration` is returned *before* the round/clamp: those are integer ops whose
    result decides a shape, so they belong to the host, and keeping them out of
    the graph is what makes this half static.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def textencoder(self, input_ids):
        """`TextEncoder.forward` without the packing, and without the masking.

        The masks are `masked_fill_(m, 0)` where `m` is all-false at exact
        length — provably no-ops here, and dropping them keeps the graph honest
        about what it computes rather than emitting a `where` that can never
        fire.
        """
        te = self.model.text_encoder
        x = te.embedding(input_ids).transpose(1, 2)
        for c in te.cnn:
            x = c(x)
        x = x.transpose(1, 2)
        x = lstm(te.lstm, x)
        return x.transpose(-1, -2)

    def durationencoder(self, x, style):
        """`DurationEncoder.forward` without the packing."""
        de = self.model.predictor.text_encoder
        from kokoro.modules import AdaLayerNorm
        x = x.permute(2, 0, 1)
        s = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, s], axis=-1)
        x = x.transpose(0, 1).transpose(-1, -2)
        for block in de.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, s.permute(1, 2, 0)], axis=1)
            else:
                x = lstm(block, x.transpose(-1, -2)).transpose(-1, -2)
        return x.transpose(-1, -2)

    def forward(self, input_ids, ref_s):
        m = self.model
        s = ref_s[:, 128:]
        # `attention_mask` all ones: exact length, nothing padded.
        mask = torch.ones_like(input_ids)
        with exact():                     # see `exact` — bert costs 0.962 in fp16
            bert_dur = m.bert(input_ids, attention_mask=mask)
        d_en = m.bert_encoder(bert_dur).transpose(-1, -2)
        d = self.durationencoder(d_en, s)
        x = lstm(m.predictor.lstm, d)
        duration = torch.sigmoid(m.predictor.duration_proj(x)).sum(axis=-1)
        t_en = self.textencoder(input_ids)
        return d, t_en, duration


class VocoderHalf(torch.nn.Module):
    """`(en, asr, s) -> audio`.

    `en` and `asr` are the aligned, frame-rate tensors the host produced by
    gathering `d` and `t_en` through the expanded durations. Static in the frame
    count.

    This is where the expensive sequential work is: `predictor.shared` is a
    bidirectional LSTM over *frames*, not tokens, and everything after it is
    convolution.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, en, asr, ref_s):
        # TWO different style halves, and they are not interchangeable:
        # `ref_s[:, 128:]` conditions the prosody predictor, `ref_s[:, :128]` the
        # decoder. Passing one for both is silent — it produces speech-shaped
        # audio at rel rms 1.3 from the reference, which is a different voice
        # rather than a broken one.
        p = self.model.predictor
        sp, sd = ref_s[:, 128:], ref_s[:, :128]
        x = lstm(p.shared, en.transpose(-1, -2))
        # F0 in fp32 even under autocast: its error accumulates PHASE through the
        # excitation's `sin(cumsum(F0))`, so it is the one block whose fp16 loss
        # grows with the utterance. See `exact`.
        with exact():
            F0 = x.transpose(-1, -2)
            for block in p.F0:
                F0 = block(F0, sp)
            F0 = p.F0_proj(F0)
        N = x.transpose(-1, -2)
        for block in p.N:
            N = block(N, sp)
        N = p.N_proj(N)
        return self.model.decoder(asr, F0.squeeze(1), N.squeeze(1), sd).squeeze()


def symbolize(gs, vs, sym):
    """Recover a dynamic axis from THREE exports at different lengths.

    `torch.export` refuses to keep the sequence length symbolic through an LSTM —
    `aten::lstm`'s meta implementation specialises it, so `dynamic_shapes` fails
    with "you marked T as dynamic but your code specialized it to be a constant"
    even when the op is called directly. And **padding to a bucket cannot stand in
    for it**: measured on this model, +2 pad tokens changes the predicted
    durations (98 -> 95 frames) and gives cosine **-0.09** against the exact run,
    because the LSTMs are bidirectional and one pad token at the end changes every
    earlier state.

    So the dynamism is recovered afterwards. Export at several lengths; any
    integer that differs between them is a function of the axis.

    **Three lengths, not two, and the third is a check rather than a fit.** Two
    points always determine a line, and most axes here are linear — but not all:
    the ALBERT block pads its sequence to a multiple of 8, so one axis reads 32 at
    both t=30 and t=31 and then 56 at t=55. A two-point fit calls that a constant
    and the graph dies at another length with a `BoundsError` naming neither the
    axis nor the padding. Every candidate is therefore verified against the third
    point, and anything that survives none of the forms is an error rather than a
    guess.

    `DNNKernels` evaluates shapes as Julia expressions against the `dims` passed
    to `call` (SAM 2's graphs carry `16*h`), and its `evalexpr` understands
    `ceiling(a, b)` as `cld`.
    """
    v0, v1, v2 = vs

    def fit(a0, a1, a2):
        """An expression matching all three, or None."""
        if a0 == a1 == a2:
            return a0
        k, r = divmod(a1 - a0, v1 - v0)
        if r == 0:
            c = a0 - k * v0
            if k * v2 + c == a2:                       # linear, verified
                if k == 0:
                    return c
                term = sym if k == 1 else f"{k}*{sym}"
                return term if c == 0 else f"{term} + {c}" if c > 0 else f"{term} - {-c}"
        # Padded to a multiple: `k*m*ceil(v/m) + j*v + c`. The `j` term is not
        # decoration — a `constant_pad_nd` stores the pad AMOUNT, which is
        # `8*ceil(t/8) - t` and reads 2/1/1 at lengths 30/31/55: a ceiling and a
        # linear term together, and neither alone fits it.
        for m in (2, 4, 8, 16, 32, 64, 128):
            for k in (1, 2, 4):
                for j in (0, -1, 1, -2, 2):
                    f = lambda v: k * m * -(-v // m) + j * v
                    c = a0 - f(v0)
                    if f(v1) + c == a1 and f(v2) + c == a2:
                        term = f"{k * m}*ceiling({sym}, {m})"
                        if j == 1:
                            term += f" + {sym}"
                        elif j == -1:
                            term += f" - {sym}"
                        elif j != 0:
                            term += f" + {j}*{sym}" if j > 0 else f" - {-j}*{sym}"
                        return term if c == 0 else f"{term} + {c}" if c > 0 else f"{term} - {-c}"
        return None

    unresolved = []

    def walk(path, xs):
        a0, a1, a2 = xs
        if isinstance(a0, bool):
            return a0
        if all(isinstance(v, int) for v in xs):
            e = fit(a0, a1, a2)
            if e is None:
                unresolved.append((path, a0, a1, a2))
                return a0
            return e
        if all(isinstance(v, list) for v in xs) and len({len(v) for v in xs}) == 1:
            return [walk(f"{path}[{i}]", t) for i, t in enumerate(zip(*xs))]
        if all(isinstance(v, dict) for v in xs) and len({tuple(sorted(v)) for v in xs}) == 1:
            return {k: walk(f"{path}.{k}", tuple(v[k] for v in xs)) for k in a0}
        return a0

    out = json.loads(json.dumps(gs[0]))
    others = [{x["id"]: x for x in g["buffers"]} for g in gs[1:]]
    n = 0
    for buf in out["buffers"]:
        for key in ("shape", "attrs"):
            if key not in buf:
                continue
            new = walk(f"{buf['id']}.{key}", (buf[key], *[o[buf["id"]][key] for o in others]))
            n += new != buf[key]
            buf[key] = new
    oo = [{o["id"]: o for o in g["ops"]} for g in gs[1:]]
    for op in out["ops"]:
        new = walk(f"{op['id']}.attrs",
                   (op.get("attrs", {}), *[o[op["id"]].get("attrs", {}) for o in oo]))
        n += new != op.get("attrs", {})
        op["attrs"] = new
    if unresolved:
        lines = "\n".join(f"    {p}: {a}/{b}/{c} at lengths {vs}" for p, a, b, c in unresolved[:8])
        raise RuntimeError(
            f"{len(unresolved)} integers move with `{sym}` in a form that is neither "
            f"linear nor a pad-to-multiple:\n{lines}\n"
            "Guessing one would produce a graph that is wrong at untested lengths.")
    out["symbols"] = [sym]
    return out, n


def align(duration, ntokens, nframes=0, speed=1.0):
    """Durations -> a per-frame token index, on the host.

    `pred_dur[i]` frames of token `i`, concatenated. `nframes = 0` means the true
    length, which is the only one that is correct.

    **Padding this to a fixed frame count does not work**, and the parity check
    catches it: `predictor.shared` is a *bidirectional* LSTM over frames, so a
    reverse pass over a padded tail starts inside the padding and contaminates
    every real frame before it. Measured — padding 98 true frames out to 640 gave
    rel rms **2.6** against the reference, i.e. different audio, not slightly
    worse audio. It is the same objection that rules out padding on the token
    side, one axis down.
    """
    dur = torch.round(duration / speed).clamp(min=1).long().squeeze()
    idx = torch.repeat_interleave(torch.arange(ntokens, device=dur.device), dur)
    n = int(idx.shape[0])
    nframes <= 0 and (nframes := n)
    if n < nframes:
        idx = torch.cat([idx, idx.new_full((nframes - n,), ntokens - 1)])
    return idx[:nframes], min(n, nframes)


def checkparity(model, text, voc, ids, ref_s, nframes):
    """Assert the split reproduces `forward_with_tokens` before writing anything.

    The two halves rewrite `TextEncoder` and `DurationEncoder` by hand to remove
    the packing, and a hand-rewritten recurrent stack that is subtly wrong still
    produces speech-shaped audio. So the whole path is checked against the real
    model on the same tokens.
    """
    ntokens = ids.shape[1]
    # SEEDED, both sides. `SineGen` draws `torch.rand` for the initial phase and
    # `torch.randn_like` for the noise floor of its harmonic-plus-noise source,
    # so the vocoder is stochastic: two calls on identical inputs differ, and an
    # unseeded comparison measures the noise and nothing else.
    torch.manual_seed(0)
    want, wantdur = model.forward_with_tokens(ids, ref_s, 1.0)

    d, t_en, duration = text(ids, ref_s)
    idx, ntrue = align(duration, ntokens, nframes)
    en = d.transpose(-1, -2)[:, :, idx]
    asr = t_en[:, :, idx]
    torch.manual_seed(0)
    got = voc(en, asr, ref_s)

    dur = torch.round(duration).clamp(min=1).long().squeeze()
    assert torch.equal(dur, wantdur), "durations differ from the reference"
    n = min(want.shape[-1], got.shape[-1])
    d_ = (got[:n] - want[:n]).float()
    rel = (d_.pow(2).mean().sqrt() / want[:n].float().pow(2).mean().sqrt()).item()
    print(f"  parity vs forward_with_tokens: {ntrue} frames, "
          f"{n} samples, rel rms {rel:.3e}")
    assert rel < 1e-4, f"the split diverges: rel rms {rel:.3e}"


def precision_ctx(precision):
    """The dtype policy the graph is traced under.

    `torch.export` captures autocast as explicit `_to_copy` nodes, so the dtype of
    every buffer becomes data in the JSON rather than a runtime policy the Julia
    side would have to reimplement — **provided `run_decompositions()` runs inside
    the same context.** Called outside, it re-traces and silently drops every
    cast, and the graph comes back fp32 with no error to say so.
    """
    if precision == "autocast":
        return torch.amp.autocast(device_type="cuda", enabled=True)
    return contextlib.nullcontext()


def main(out: Path, dev: str, ntokens: int, nframes: int, phonemes: str,
         precision: str = "fp32"):
    from kokoro import KModel
    model = KModel(repo_id="hexgrad/Kokoro-82M",
                   config=str(WEIGHTS / "config.json"),
                   model=str(WEIGHTS / "kokoro-v1_0.pth")).eval().to(dev)

    ids = torch.tensor([[0] + [model.vocab[c] for c in phonemes if c in model.vocab] + [0]],
                       device=dev)
    if ids.shape[1] != ntokens:
        print(f"  phoneme string gives {ids.shape[1]} tokens; exporting at that length")
        ntokens = ids.shape[1]
    # A Kokoro voice is `(511, 1, 256)`: one style vector per token count, so the
    # prosody is conditioned on how long the utterance is. `KPipeline` indexes it
    # `pack[len(ps) - 1]`, which is already `(1, 256)` — no unsqueeze.
    voice = torch.load(WEIGHTS / "af_heart.pt", map_location="cpu", weights_only=True)
    ref_s = voice[ids.shape[1] - 1].to(dev)

    # The generator's STFT stays fp32 — see `keepexact`. Under autocast its
    # inverse goes complex32, which nothing downstream can load.
    if precision == "autocast":
        keepexact(model.decoder.generator.stft, "transform", "inverse")

    text = TextHalf(model).eval()
    voc = VocoderHalf(model).eval()

    with torch.no_grad():
        if nframes <= 0:                 # the utterance's own frame count
            _, _, dur0 = text(ids, ref_s)
            _, nframes = align(dur0, ids.shape[1])
            print(f"  durations give {nframes} frames; exporting at that length")
        checkparity(model, text, voc, ids, ref_s, nframes)

        out.mkdir(parents=True, exist_ok=True)
        available = dict(text.state_dict())
        available.update(dict(text.named_buffers()))
        tensors = {}

        def emit(mod, args, name, more=None, vs=None, sym=""):
            def conv(a):
                with precision_ctx(precision):
                    ep = torch.export.export(mod, a, strict=False)
                    # INSIDE the context — see `precision_ctx`.
                    ep = ep.run_decompositions(decomptable())
                # Lifted constants come off the ExportedProgram, so they are
                # collected here rather than from a variable the caller can see.
                available.update({k: v for k, v in ep.constants.items()
                                  if isinstance(v, torch.Tensor)})
                return EG.convert(ep, ({},), name)
            g = conv(args)
            if more is not None:
                gs = [g] + [conv(a) for a in more]
                ref = [o["id"] for o in g["ops"]]
                for k, gg in enumerate(gs[1:], 1):
                    if [o["id"] for o in gg["ops"]] != ref:
                        raise RuntimeError(f"{name}: length {vs[k]} gave a different op order; "
                                           "the graph structure is not length-invariant")
                g, nsym = symbolize(gs, vs, sym)
                print(f"  {name}: {nsym} shapes/attrs became functions of `{sym}` "
                      f"(fitted at {vs[0]}/{vs[1]}, verified at {vs[2]})")
            live = set(g["inputs"]) | set(g["outputs"])
            for o in g["ops"]:
                live.update(o["in"]); live.add(o["out"])
            for b in g["buffers"]:
                b.get("of") and live.add(b["of"])
            keep = [b for b in g["buffers"]
                    if b.get("kind") != "weight" or b["id"] in live]
            g["buffers"] = keep
            for b in keep:
                k = b.get("key")
                if b.get("kind") == "weight" and k is not None and k not in tensors:
                    if k not in available:
                        raise KeyError(f"{name} wants {k}, which is in no state dict")
                    tensors[k] = available[k].detach().contiguous().cpu()
            (out / f"{name}.json").write_text(json.dumps(g, indent=1))
            hist = {}
            for o in g["ops"]:
                hist[o["aten"]] = hist.get(o["aten"], 0) + 1
            print(f"  {name}: {len(g['ops'])} ops, {len(keep)} buffers")
            return hist

        # A second export one token longer, purely to diff against. `ref_s` is
        # indexed by token count in the real pipeline, but here it is only an
        # input whose own shape does not move, so the same one serves both.
        # Three lengths: two to fit, one well away from them to verify. The
        # third catches pad-to-a-multiple axes, which two adjacent lengths cannot
        # distinguish from a constant.
        tv = (ntokens, ntokens + 1, ntokens + 25)
        grow = lambda n: torch.cat([ids] + [ids[:, -1:]] * (n - ntokens), dim=1)
        h1 = emit(text, (ids, ref_s), "kokorotext",
                  more=[(grow(tv[1]), ref_s), (grow(tv[2]), ref_s)], vs=tv, sym="t")
        d, t_en, duration = text(ids, ref_s)
        idx, _ = align(duration, ntokens, nframes)
        en = d.transpose(-1, -2)[:, :, idx].contiguous()
        asr = t_en[:, :, idx].contiguous()
        fv = (nframes, nframes + 1, nframes + 37)
        wide = lambda x, n: torch.cat([x] + [x[:, :, -1:]] * (n - nframes), dim=2).contiguous()
        h2 = emit(voc, (en, asr, ref_s), "kokorovoc",
                  more=[(wide(en, fv[1]), wide(asr, fv[1]), ref_s),
                        (wide(en, fv[2]), wide(asr, fv[2]), ref_s)], vs=fv, sym="f")

    save_file(tensors, str(out / "weights.safetensors"))
    hist = {k: h1.get(k, 0) + h2.get(k, 0) for k in set(h1) | set(h2)}
    (out / "op_histogram.json").write_text(json.dumps(hist, indent=1, sort_keys=True))
    nb = sum(v.numel() * v.element_size() for v in tensors.values())
    print(f"kokoro: {ntokens} tokens, {nframes} frames, "
          f"{sum(v.numel() for v in tensors.values()) / 1e6:.1f}M params, {nb / 2**20:.0f} MiB")
    for k, v in sorted(hist.items(), key=lambda kv: -kv[1])[:14]:
        print(f"  {v:5d}  {k}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=GEN / "graphs" / "kokoro")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tokens", type=int, default=0, help="0 = whatever the phonemes give")
    p.add_argument("--frames", type=int, default=0,
                   help="0 = the frame count the durations produce (the only correct one)")
    p.add_argument("--phonemes", default="hˈEllO wˈɜɹld, ðɪs ɪz kˈOkəɹO",
                   help="already phonemised; KPipeline does text -> phonemes")
    p.add_argument("--precision", default="fp32", choices=["fp32", "autocast"],
                   help="autocast puts 88%% of the weights in fp16; see `exact`")
    a = p.parse_args()
    main(a.out, a.device, a.tokens, a.frames, a.phonemes, a.precision)
