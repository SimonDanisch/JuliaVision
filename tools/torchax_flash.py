"""A fused-attention lowering for torchax, so XLA can run the graph as traced.

Importing this module registers `aten._scaled_dot_product_flash_attention` with
torchax. Without it that op has no lowering at all — torchax ships 434 aten ops
and not one attention — so the only way to get SAM 2's encoder through torchax
was `--sdpa math`, which decomposes attention into a materialised softmax and is
a different algorithm from the one Lava and PyTorch run.

**What it is worth**, measured on this device (Radeon 8060S, gfx1151), SAM 2.1
large encode at 1024x1024, every row parity-gated against the refs:

    PyTorch compiled, fused           139.9 ms
    PyTorch eager, fused              174.0 ms
    XLA with this lowering            176.4 ms
    Lava                              272.65 ms
    XLA, math attention               279.5 ms
    PyTorch eager, math attention     451.4 ms

XLA was already 1.62x faster than PyTorch on the *same* math graph, so it never
had a codegen problem — the whole of its deficit against fused PyTorch was the
attention algorithm, and closing that put it level with eager (176.4 against
174.0) and 1.55x ahead of Lava. One `(1, 4096, 8, 72)` attention costs 8.89 ms
through `jax.nn.dot_product_attention` (whose only implementation here is `xla`;
`cudnn` is CUDA-only) against 3.79 ms through Pallas.

Checked per shape against `F.scaled_dot_product_attention` before being trusted
in aggregate: at the encoder's six real attention shapes, all head dim 72, this
agrees to 6.1e-05 .. 9.8e-04, which is fp16 rounding. Worth doing that way
round, because `add_129` — the one output whose drift is pinned rather than
bounded — moves with the runtime and not with the algorithm, so it cannot tell a
wrong kernel from a differently-accumulated one.

`jax.experimental.pallas.ops.gpu.attention` warns that it is deprecated in
favour of `tokamax.dot_product_attention`. That is the next thing to try here,
both for longevity and because it may be quicker.

**Small sequences stay on the math path.** Pallas' `mha` blocks the sequence,
and hiera's windowed attention runs sequence lengths well under one block. Below
`MINSEQ` this defers to `jax.nn.dot_product_attention`, which is what those
shapes were going to get anyway; the global-attention blocks, which are the ones
that cost, take the fused path. A run is therefore not uniformly "flash" and
should not be described as such.
"""

import jax
import jax.numpy as jnp
import torch
from jax.experimental.pallas.ops.gpu import attention as pallas_attention

from torchax.ops.jaten import op

# One Pallas block. Shorter sequences than this get no benefit from the tiled
# kernel and can trip its block constraints, so they take the other path.
MINSEQ = 128


@op(torch.ops.aten._scaled_dot_product_flash_attention)
def _flash_attention(query, key, value, dropout_p=0.0, is_causal=False,
                     return_debug_mask=False, *, scale=None):
    # torch hands attention `(B, H, S, D)`; Pallas wants `(B, S, H, D)`.
    q, k, v = (jnp.swapaxes(t, 1, 2) for t in (query, key, value))
    dropout_p == 0.0 or _unsupported("dropout", dropout_p)
    sm_scale = float(scale) if scale is not None else q.shape[-1] ** -0.5

    if q.shape[1] >= MINSEQ and k.shape[1] >= MINSEQ:
        out = pallas_attention.mha(q, k, v, segment_ids=None, sm_scale=sm_scale,
                                   causal=bool(is_causal))
    else:
        # In fp32, and cast back. The graph's own decomposition of this op keeps
        # the softmax in fp32 — that is what autocast's `_to_copy` nodes say —
        # and doing it in fp16 here instead showed up 543 ops later as `add_129`
        # drifting to 1.005 against the references, four times what the same
        # graph drifts through XLA's own decomposition.
        out = jax.nn.dot_product_attention(
            q.astype(jnp.float32), k.astype(jnp.float32), v.astype(jnp.float32),
            scale=sm_scale, is_causal=bool(is_causal), implementation="xla"
        ).astype(q.dtype)
    out = jnp.swapaxes(out, 1, 2)

    # The schema returns nine values and the graph reads only the first: every
    # trace of this op has `num_users=1` through a `getitem 0`. The rest are
    # shaped plausibly rather than left as `None` so that anything which does
    # look at them sees an array of the right rank instead of a type error.
    b, h, s, _ = out.shape
    lse = jnp.zeros((b, h, s), jnp.float32)
    zero = jnp.zeros((), jnp.int32)
    return (out, lse, zero, zero, 0, 0, zero, zero, jnp.zeros((), out.dtype))


def _unsupported(what, value):
    raise NotImplementedError(
        f"fused attention lowering got {what}={value}; this exists for "
        "inference graphs, where it is always off")
