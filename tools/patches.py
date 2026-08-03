"""Trace-friendly rewrites of three upstream constructs, and their equivalence proof.

None of these change the arithmetic. Each replaces a formulation torch.export
cannot represent statically with one it can, and `verify()` checks each rewrite
against the original on real tensors before any graph is emitted.

1. interpolate_groups / UpsampleBlock (group_modules.py:7, modules.py:16)
   Use `scale_factor=`, which PyTorch resolves through float math, so export
   emits a guard `w == trunc(0.5*float(2*w))` it cannot discharge and the
   `segment` graph fails to export at all. Rewritten to integer `size=`.
   For exact power-of-two factors the resolved scale is identical, so the
   sampled coordinates are identical.

2. QueryTransformer._get_aux_mask (object_transformer.py:204)
   `aux_mask[torch.where(aux_mask.sum(-1) == aux_mask.shape[-1])] = False`
   lowers to nonzero + index_put: a data-dependent output size, which cannot
   be allocated ahead of time. It is a degenerate-row guard - a row that masks
   every position would make the attention softmax produce NaN - and it is
   expressible elementwise as `mask & ~mask.all(-1, keepdim=True)`, which has
   static shape.

   This is not the "guard depends on tensor contents" escalation in
   lava-dnn.md: no control flow and no shape depends on it. Only values do.
"""

import torch
import torch.nn.functional as F


def interpolate_groups(g, ratio, mode, align_corners):
    """group_modules.py:7, with integer sizes."""
    batch_size, num_objects = g.shape[:2]
    x = g.flatten(start_dim=0, end_dim=1)
    h, w = x.shape[-2:]
    if ratio >= 1:
        size = (h * int(ratio), w * int(ratio))
    else:
        inv = int(round(1 / ratio))
        size = (h // inv, w // inv)
    x = F.interpolate(x, size=size, mode=mode, align_corners=align_corners)
    return x.view(batch_size, num_objects, *x.shape[1:])


def upsample_block_forward(self, in_g, skip_f):
    """modules.py:16, with integer sizes."""
    h, w = in_g.shape[-2:]
    g = F.interpolate(in_g, size=(h * int(self.scale_factor), w * int(self.scale_factor)),
                      mode="bilinear")
    g = self.out_conv(g)
    return g + skip_f


def get_aux_mask(self, logits, selector=None, seg_pass=False):
    """object_transformer.py:180, with the degenerate-row fix done elementwise."""
    from matanyone2.utils.tensor_utils import aggregate

    prob = logits.sigmoid() if selector is None else logits.sigmoid() * selector
    logits = aggregate(prob, dim=1)

    is_foreground = (logits[:, 1:] >= logits.max(dim=1, keepdim=True)[0])
    foreground_mask = is_foreground.bool().flatten(start_dim=2)

    aux_foreground_mask = (~foreground_mask).unsqueeze(2).unsqueeze(2).repeat(
        1, 1, self.num_heads, self.num_queries // 2, 1).flatten(start_dim=0, end_dim=2)
    aux_background_mask = foreground_mask.unsqueeze(2).unsqueeze(2).repeat(
        1, 1, self.num_heads, self.num_queries // 2, 1).flatten(start_dim=0, end_dim=2)

    aux_mask = torch.cat([aux_foreground_mask, aux_background_mask], dim=1)

    # a row that blocks every position is un-blocked entirely; upstream writes
    # this with fancy indexing, which needs nonzero
    return aux_mask & ~aux_mask.all(dim=-1, keepdim=True)


def apply():
    """Install the rewrites. Call before building any wrapper module."""
    from matanyone2.model import group_modules, modules
    from matanyone2.model.modules import UpsampleBlock
    from matanyone2.model.transformer.object_transformer import QueryTransformer

    group_modules.interpolate_groups = interpolate_groups
    # modules.py imported the names directly, so rebind there too
    modules.upsample_groups = lambda g, ratio=2, mode="bilinear", align_corners=False: \
        interpolate_groups(g, ratio, mode, align_corners)
    modules.downsample_groups = lambda g, ratio=1 / 2, mode="area", align_corners=None: \
        interpolate_groups(g, ratio, mode, align_corners)
    UpsampleBlock.forward = upsample_block_forward
    QueryTransformer._get_aux_mask = get_aux_mask


def clear_pe_caches(model):
    """Drop cached positional encodings before exporting.

    positional_encoding.py:65 returns the cache when `cached_penc.shape ==
    tensor.shape`. If the model has already run at some resolution, that
    comparison becomes a guard that specializes h and w, and export fails with
    "you marked 16*h as dynamic but your code specialized it". Harmless to call
    always; the encoding is recomputed in-graph from arange/sin/cos.
    """
    n = 0
    for m in model.modules():
        if getattr(m, "cached_penc", None) is not None:
            m.cached_penc = None
            n += 1
    return n


def verify(device="cuda", verbose=True):
    """Check every rewrite against the original. Returns list of (name, max_abs_diff)."""
    import importlib

    import matanyone2.model.group_modules as gm
    import matanyone2.model.modules as md
    from matanyone2.model.transformer import object_transformer as ot

    # reload to get pristine originals even if apply() already ran
    importlib.reload(gm)
    importlib.reload(md)
    importlib.reload(ot)

    results = []
    torch.manual_seed(0)

    for ratio, mode, ac in [(2, "bilinear", False), (1 / 2, "area", None),
                            (1 / 4, "area", None), (1 / 8, "area", None),
                            (1 / 16, "area", None)]:
        g = torch.randn(1, 1, 8, 64, 96, device=device)
        a = gm.interpolate_groups(g, ratio, mode, ac)
        b = interpolate_groups(g, ratio, mode, ac)
        assert a.shape == b.shape, f"shape {a.shape} vs {b.shape} for ratio {ratio}"
        results.append((f"interpolate_groups(ratio={ratio}, {mode})",
                        (a - b).abs().max().item()))

    blk = md.UpsampleBlock(16, 16).to(device).eval()
    in_g = torch.randn(1, 16, 30, 40, device=device)
    skip = torch.randn(1, 16, 60, 80, device=device)
    with torch.no_grad():
        a = md.UpsampleBlock.forward(blk, in_g, skip)
        b = upsample_block_forward(blk, in_g, skip)
    results.append(("UpsampleBlock.forward", (a - b).abs().max().item()))

    # aux mask: random logits, plus a case engineered to hit the degenerate row
    class Fake:
        num_heads, num_queries = 8, 16

    for name, logits in [
        ("aux_mask/random", torch.randn(1, 1, 12, 20, device=device)),
        ("aux_mask/all-foreground", torch.full((1, 1, 12, 20), 9.0, device=device)),
        ("aux_mask/all-background", torch.full((1, 1, 12, 20), -9.0, device=device)),
    ]:
        a = ot.QueryTransformer._get_aux_mask(Fake(), logits, None)
        b = get_aux_mask(Fake(), logits, None)
        assert a.shape == b.shape
        results.append((name, float((a != b).sum().item())))

    if verbose:
        for n, d in results:
            print(f"  {'OK ' if d == 0 else 'DIFF'} {n:44} {d}")
    return results


if __name__ == "__main__":
    import common

    common.bootstrap()
    res = verify()
    bad = [r for r in res if r[1] != 0]
    print(f"\n{len(res)} rewrites checked, {len(bad)} differ" + (f": {bad}" if bad else " - all bit-exact"))
    raise SystemExit(1 if bad else 0)
