"""
`WanModel.forward`, rewritten so it exports.

The stock forward takes *lists* of variable-length latents and text embeddings
and pads each up to `seq_len`/`text_len`:

    seq_lens = torch.tensor([u.size(1) for u in x])
    x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), ...)]) ...])

`torch.export` turns those Python-level sizes into *unbacked* symbols and then
cannot decide a comparison on them, failing with
`GuardOnDataDependentSymNode: Could not guard on data-dependent expression
22*u4 < 2`. The padding is also dead weight for a single clip whose length is
already `seq_len`.

`StaticWan` runs the same computation for exactly one video and one prompt, with
every length a Python constant. It calls the model's own submodules — patch
embedding, time embedding, the 30 blocks, the head — so the weights and the
arithmetic are unchanged; only the list plumbing is gone.

Verified against the stock forward, not just exported: `check()` runs both and
reports the difference.
"""

import math

import torch
import torch.nn as nn


class StaticWan(nn.Module):
    """One clip, one prompt, all shapes fixed."""

    def __init__(self, model, grid, seq_len: int):
        super().__init__()
        self.m = model
        self.grid = tuple(int(g) for g in grid)      # (F, H/p, W/p) after patching
        self.seq_len = int(seq_len)
        # Built ONCE, as plain attributes. `rope_apply` does
        # `grid_sizes[i].tolist()`, and a tensor constructed inside `forward` is
        # traced — so `.tolist()` yields *unbacked* symints and export dies on
        # `GuardOnDataDependentSymNode`. Created here they are constants and the
        # rotary path folds away.
        self._seq_lens = torch.tensor([self.seq_len], dtype=torch.long)
        self._grid_sizes = torch.tensor([list(self.grid)], dtype=torch.long)
        assert math.prod(self.grid) == self.seq_len, \
            f"grid {self.grid} has {math.prod(self.grid)} tokens, seq_len is {seq_len}"

    def forward(self, x, t, context):
        """x: (C_in, F, H, W) · t: (1,) · context: (text_len, text_dim)."""
        m = self.m
        # patchify — the stock code does this per list element
        u = m.patch_embedding(x.unsqueeze(0))                    # (1, dim, f, h, w)
        u = u.flatten(2).transpose(1, 2)                         # (1, L, dim)

        # time embedding. `seq_lens`/`grid_sizes` are constants here, so the
        # tensors the blocks want are built once rather than derived from shapes.
        tt = t.expand(t.size(0), self.seq_len).flatten()
        e = m.time_embedding(
            sinusoidal(m.freq_dim, tt).unflatten(0, (t.size(0), self.seq_len)).float())
        e0 = m.time_projection(e).unflatten(2, (6, m.dim))

        ctx = m.text_embedding(context.unsqueeze(0))

        for block in m.blocks:
            u = block(u, e=e0, seq_lens=self._seq_lens, grid_sizes=self._grid_sizes,
                      freqs=m.freqs, context=ctx, context_lens=None)
        u = m.head(u, e)
        return unpatchify(u, self.grid, m.patch_size, m.out_dim)


def sinusoidal(dim, position):
    """`sinusoidal_embedding_1d`, inlined so the wrapper has no import cycle."""
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1).float()


def unpatchify(u, grid, patch_size, out_dim):
    """The model's `unpatchify` for a single clip, with constant extents."""
    f, h, w = grid
    pf, ph, pw = patch_size
    u = u[0].view(f, h, w, pf, ph, pw, out_dim)
    u = torch.einsum("fhwpqrc->cfphqwr", u)
    return u.reshape(out_dim, f * pf, h * ph, w * pw).unsqueeze(0)


def check(model, static, x, t, context, seq_len):
    """Stock forward vs the static one, on the same inputs."""
    with torch.no_grad():
        a = model([x], t, [context], seq_len=seq_len)[0]
        b = static(x, t, context)[0]
    d = (a - b).abs()
    return float(d.max()), tuple(a.shape), tuple(b.shape)
