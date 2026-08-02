"""Wrapper modules for the pieces InferenceCore drives, and their example inputs.

One wrapper per entry in enumerate.py's TRACED list. Each holds only the
submodules it needs, so torch.export lifts only the parameters that graph
actually uses. Shapes are given at stride-16 h,w and full-resolution H,W with
H = 16h, W = 16w; both are dynamic, resolution is not a variant axis.

The memory read (get_similarity / do_softmax / readout) is deliberately absent:
it is the one graph whose length depends on frame index rather than resolution,
so it is authored against a padded fixed capacity instead of traced. See
memory_read.py.
"""

import torch
import torch.nn as nn


class EncodeImage(nn.Module):
    """image -> (f16, f8, f4, f2, f1), pix_feat.  matanyone2.py:88"""

    def __init__(self, m):
        super().__init__()
        self.pixel_encoder = m.pixel_encoder
        self.pix_feat_proj = m.pix_feat_proj
        self.register_buffer("pixel_mean", m.pixel_mean.clone())
        self.register_buffer("pixel_std", m.pixel_std.clone())

    def forward(self, image):
        image = (image - self.pixel_mean) / self.pixel_std
        ms = self.pixel_encoder(image, None)
        return (*ms, self.pix_feat_proj(ms[0]))


class TransformKey(nn.Module):
    """f16 -> key, shrinkage, selection.  matanyone2.py:117"""

    def __init__(self, m):
        super().__init__()
        self.key_proj = m.key_proj

    def forward(self, f16):
        return self.key_proj(f16, need_s=True, need_e=True)


class EncodeMask(nn.Module):
    """image, pix_feat, sensory, masks -> mask_value, sensory, obj_summary.

    matanyone2.py:95. deep_update is a construction-time flag, not an input:
    it picks a branch inside MaskEncoder, so the two settings are two graphs.
    """

    def __init__(self, m, deep_update):
        super().__init__()
        self.mask_encoder = m.mask_encoder
        self.object_summarizer = m.object_summarizer
        self.deep_update = deep_update
        self.register_buffer("pixel_mean", m.pixel_mean.clone())
        self.register_buffer("pixel_std", m.pixel_std.clone())

    def forward(self, image, pix_feat, sensory, masks):
        image = (image - self.pixel_mean) / self.pixel_std
        # single_object: _get_others returns None (matanyone2.py:63)
        mask_value, new_sensory = self.mask_encoder(
            image, pix_feat, sensory, masks, None,
            deep_update=self.deep_update, chunk_size=-1)
        summaries, _ = self.object_summarizer(masks, mask_value, False)
        return mask_value, new_sensory, summaries


class PixelFusion(nn.Module):
    """pix_feat, pixel, sensory, last_mask -> fused.  matanyone2.py:203"""

    def __init__(self, m):
        super().__init__()
        self.pixel_fuser = m.pixel_fuser

    def forward(self, pix_feat, pixel, sensory, last_mask):
        import torch.nn.functional as F

        last_mask = F.interpolate(last_mask, size=sensory.shape[-2:], mode="area")
        return self.pixel_fuser(pix_feat, pixel, sensory, last_mask, None, chunk_size=-1)


class ReadoutQuery(nn.Module):
    """pixel_readout, obj_memory -> mem_readout.  matanyone2.py:220"""

    def __init__(self, m):
        super().__init__()
        self.object_transformer = m.object_transformer

    def forward(self, pixel_readout, obj_memory):
        out, _ = self.object_transformer(pixel_readout, obj_memory,
                                         selector=None, need_weights=False, seg_pass=False)
        return out


class PredUncertainty(nn.Module):
    """last_pix_feat, pix_feat, last_mask, mem_val_diff -> prob.  matanyone2.py:73"""

    def __init__(self, m):
        super().__init__()
        self.temp_sparity = m.temp_sparity

    def forward(self, last_pix_feat, pix_feat, last_mask, mem_val_diff):
        logits = self.temp_sparity(last_frame_feat=last_pix_feat, cur_frame_feat=pix_feat,
                                   last_mask=last_mask, mem_val_diff=mem_val_diff)
        return torch.sigmoid(logits)


class Segment(nn.Module):
    """ms_feat, memory_readout, sensory -> sensory, prob.  matanyone2.py:233

    clamp_mat=True and seg_pass=False are fixed on the matting path.
    """

    def __init__(self, m, update_sensory=True):
        super().__init__()
        self.mask_decoder = m.mask_decoder
        self.update_sensory = update_sensory

    def forward(self, f16, f8, f4, f2, f1, memory_readout, sensory):
        sensory, logits = self.mask_decoder(
            [f16, f8, f4, f2, f1], memory_readout, sensory,
            chunk_size=-1, update_sensory=self.update_sensory,
            seg_pass=False, last_mask=None, sigmoid_residual=False)
        logits = logits.clamp(0.0, 1.0)
        prob = torch.cat([torch.prod(1 - logits, dim=1, keepdim=True), logits], 1)
        return sensory, prob


def build(model, typeparams):
    """Every graph to export, with example inputs and which dims are dynamic.

    h, w are the stride-16 feature size; the full frame is 16h x 16w. Example
    values are the 720p trace (45x80) but nothing depends on them.
    """
    h, w = 45, 80
    tp = typeparams
    C = tp["value_dim"]
    N = tp["num_objects"]
    S = tp["sensory_dim"]
    P = tp["pixel_dim"]
    K = tp["key_dim"]
    Q = tp["num_queries"]
    E = tp["embed_dim"]
    ms = tp["ms_dims"]
    dev = model.pixel_mean.device

    def t(*shape):
        return torch.randn(*shape, device=dev)

    # dynamic dim specs: 'hw' = stride-16 plane, 'HW' = full res plane
    hw = {2: "h", 3: "w"}
    HW = {2: "H", 3: "W"}

    return {
        "encode_image": (EncodeImage(model), (t(1, 3, 16 * h, 16 * w),), ({2: "H", 3: "W"},)),
        "transform_key": (TransformKey(model), (t(1, ms[0], h, w),), (hw,)),
        "encode_mask_deep": (
            EncodeMask(model, True),
            (t(1, 3, 16 * h, 16 * w), t(1, P, h, w), t(1, N, S, h, w), t(1, N, 16 * h, 16 * w)),
            (HW, hw, {3: "h", 4: "w"}, {2: "H", 3: "W"}),
        ),
        "encode_mask_shallow": (
            EncodeMask(model, False),
            (t(1, 3, 16 * h, 16 * w), t(1, P, h, w), t(1, N, S, h, w), t(1, N, 16 * h, 16 * w)),
            (HW, hw, {3: "h", 4: "w"}, {2: "H", 3: "W"}),
        ),
        "pixel_fusion": (
            PixelFusion(model),
            (t(1, P, h, w), t(1, N, C, h, w), t(1, N, S, h, w), t(1, N, 16 * h, 16 * w)),
            (hw, {3: "h", 4: "w"}, {3: "h", 4: "w"}, {2: "H", 3: "W"}),
        ),
        "readout_query": (
            ReadoutQuery(model),
            (t(1, N, C, h, w), t(1, N, 1, Q, E + 1)),
            ({3: "h", 4: "w"}, {}),
        ),
        "pred_uncertainty": (
            PredUncertainty(model),
            (t(1, P, h, w), t(1, P, h, w), t(1, N, 16 * h, 16 * w), t(1, C, h, w)),
            (hw, hw, {2: "H", 3: "W"}, hw),
        ),
        "segment": (
            Segment(model),
            (t(1, ms[0], h, w), t(1, ms[1], 2 * h, 2 * w), t(1, ms[2], 4 * h, 4 * w),
             t(1, ms[3], 8 * h, 8 * w), t(1, ms[4], 16 * h, 16 * w),
             t(1, N, C, h, w), t(1, N, S, h, w)),
            (hw, {2: "h2", 3: "w2"}, {2: "h4", 3: "w4"}, {2: "h8", 3: "w8"},
             {2: "H", 3: "W"}, {3: "h", 4: "w"}, {3: "h", 4: "w"}),
        ),
    }
