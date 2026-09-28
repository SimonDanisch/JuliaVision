"""Run the real TRELLIS.2 image-to-3D pipeline and dump every intermediate.

    ~/tools/TRELLIS.2/.venv/bin/python tools/dump_trellis2_pipeline.py --pipeline 512
    ~/tools/TRELLIS.2/.venv/bin/python tools/dump_trellis2_pipeline.py      # 1024_cascade

`export_trellis2.py` writes one graph per dense model and a reference for each.
What those references do not cover is everything between the models: the image
crop, the two DINOv3 resolutions, three flow samplers with guidance intervals and
CFG rescale, `argwhere` on the occupancy grid, the cascade's upsample and
re-quantisation, the latent normalisations, both sparse decoders and the mesh
extraction. All of it is in this file's output.

**This is a transcription of `Trellis2ImageTo3DPipeline.run`, not a
reimplementation of it.** The stage methods below are upstream's, line for line,
with `dump[...]` calls interleaved, and they drive upstream's own models and
samplers. Anything that reads like a simplification is a bug. The models are
observed through forward hooks, so no call is rerouted.

What is recorded:

| key | what |
|---|---|
| `image_rgba`, `image_pre` | the input, and `preprocess_image`'s crop (uint8 HWC) |
| `dino{512,1024}.image`, `cond{512,1024}` | DINOv3's normalised input and its output |
| `<stage>.noise`, `.steps.x`, `.steps.x0`, `.steps.v` | per-step sample, x0 and post-guidance velocity |
| `<stage>.callNN.{x,t,out}` | every model forward; `<stage>.roles` says which were `neg` |
| `ss.logits`, `ss.coords` | the 64^3 decoder output and the `argwhere` coordinates |
| `<dec>.L<i>.{coords,in,pre,sub}`, `<dec>.out` | decoder activations per level, not per block |
| `mesh.*`, `mesh_filled.*` | the extracted mesh before and after `fill_holes` |

`<dec>` is `up` (the cascade's `upsample`), `shapedec` or `texdec`. The texture
voxels upstream returns are `texdec.out * 0.5 + 0.5` on `shapedec.L4.coords`.

Stages are `ss`, then `shape512`/`shape1024` and `tex512`/`tex1024` depending on
`--pipeline`. A sparse stage stores its coordinates once as `<stage>.coords`.

Everything is saved in the dtype it was computed in. The Julia side reads the
noise from here rather than seeding its own RNG: the samplers draw on the CPU
generator, and a comparison that starts from different noise means nothing.

**Attention backends** are `export_trellis2`'s: `sdpa` for the dense models, as
the graphs are exported, and `xformers` for the sparse ones, because `flash_attn`
does not import in this venv.

Upstream: https://github.com/microsoft/TRELLIS.2
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Before torch initialises CUDA. What upstream's own example sets on 24 GB, and
# this card has 20. The attention backends are set by `export_trellis2`.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_trellis2 import CHECKOUT, GEN, REPO, bootstrap, patch_dinov3


class Dump:
    """Tensors and scalars for one run, written once at the end.

    On the CPU and in the dtype they were computed in. Booleans are stored as
    uint8, which every reader handles.
    """

    def __init__(self):
        self.tensors = {}
        self.meta = {}

    def __setitem__(self, name, value):
        if isinstance(value, torch.Tensor):
            t = value.detach()
            t = t.to(torch.uint8) if t.dtype == torch.bool else t
            assert name not in self.tensors, f"{name} recorded twice"
            self.tensors[name] = t.contiguous().cpu()
        else:
            self.meta[name] = value

    def write(self, out: Path):
        out.mkdir(parents=True, exist_ok=True)
        save_file(self.tensors, str(out / "pipeline.safetensors"))
        (out / "pipeline.json").write_text(json.dumps(self.meta, indent=1))
        total = sum(t.numel() * t.element_size() for t in self.tensors.values())
        print(f"\n{len(self.tensors)} tensors, {total / 1e6:.1f} MB -> {out}")
        big = sorted(self.tensors.items(), key=lambda kv: -kv[1].numel() * kv[1].element_size())
        for k, v in big[:25]:
            print(f"  {k:<28} {str(tuple(v.shape)):<22} {str(v.dtype).removeprefix('torch.')}")


def feats(x):
    """The tensor a sparse value carries, or the dense tensor itself."""
    from trellis2.modules.sparse import VarLenTensor
    return x.feats if isinstance(x, VarLenTensor) else x


class FlowCalls:
    """Every forward of a flow model, filed under the stage that made it.

    The sampler calls `model(x_t, t, cond, **kwargs)`; guidance runs it once
    with `cond` and once with `neg_cond`, which upstream builds as
    `zeros_like(cond)`. Which of the two a call got is decided by comparing
    against the stage's own `cond`, and a call that matches neither is an error
    rather than a guess.
    """

    def __init__(self, dump):
        self.dump = dump
        self.stage = None

    def begin(self, stage, cond, coords=None):
        self.stage, self.cond, self.coords = stage, cond, coords
        self.n, self.roles, self.concat = 0, [], None

    def end(self):
        self.dump[f"{self.stage}.roles"] = self.roles
        self.stage = None

    def __call__(self, module, args, kwargs, out):
        from trellis2.modules.sparse import SparseTensor
        assert self.stage is not None, f"{type(module).__name__} ran outside a stage"
        x, t, cond = args
        if torch.equal(cond, self.cond):
            role = "pos"
        else:
            assert cond.abs().max().item() == 0, "cond is neither the stage's nor zeros"
            role = "neg"
        if isinstance(x, SparseTensor):
            assert torch.equal(x.coords, self.coords), "coords changed inside a stage"
        cc = kwargs.get("concat_cond")
        if cc is not None:
            if self.concat is None:
                self.concat = cc.feats
                self.dump[f"{self.stage}.concat_cond"] = cc.feats
            assert torch.equal(cc.feats, self.concat), "concat_cond changed inside a stage"
        k = f"{self.stage}.call{self.n:02d}"
        self.dump[f"{k}.x"] = feats(x)
        self.dump[f"{k}.t"] = t
        self.dump[f"{k}.out"] = feats(out)
        self.roles.append(role)
        self.n += 1


class DecoderLevels:
    """A sparse decoder's activations at each resolution level.

    Per level rather than per block: the shape decoder has 32 ConvNeXt blocks and
    at 1024 its last levels carry millions of voxels, so per block is gigabytes.
    For level `i`: `L<i>.coords` and `L<i>.in` on entry, `L<i>.pre` after the
    ConvNeXt stack (the upsampling block's input), `L<i>.sub` the subdivision
    logits it predicts, and `out` after the fp32 output layer. The layer norm in
    front of that layer has no affine and is not stored.

    The texture decoder is steered by the shape decoder's `subs`, so its coords
    and subdivisions are the shape decoder's by construction. They are asserted
    equal to what `shapedec` recorded rather than stored twice.

    `rows` caps the feature rows stored per level (coords are always complete):
    at 1024 the last level is millions of voxels at 64 channels, and a prefix
    checks a decoder as well as the whole once its coords match.
    """

    def __init__(self, dump, decoder, rows=None):
        self.dump = dump
        self.rows = rows
        self.prefix = None
        self.guided = not decoder.pred_subdiv
        decoder.from_latent.register_forward_hook(self.from_latent)
        self.ups = {}
        for i, res in enumerate(decoder.blocks[:-1]):
            self.ups[id(res[-1])] = i
            res[-1].register_forward_hook(self.upsample, with_kwargs=True)
        decoder.output_layer.register_forward_hook(self.head)

    def feats(self, key, x):
        self.dump[key] = x if self.rows is None else x[:self.rows]

    def coords(self, key, c):
        if self.guided:
            want = self.dump.tensors[key.replace(self.prefix, "shapedec", 1)]
            assert torch.equal(c.cpu(), want), f"{key} differs from the shape decoder's"
        else:
            self.dump[key] = c

    def from_latent(self, module, args, out):
        if self.prefix is None:
            return
        self.coords(f"{self.prefix}.L0.coords", out.coords)
        self.feats(f"{self.prefix}.L0.in", out.feats)

    def upsample(self, module, args, kwargs, out):
        if self.prefix is None:
            return
        i = self.ups[id(module)]
        self.feats(f"{self.prefix}.L{i}.pre", args[0].feats)
        if self.guided:
            h = out
        else:
            h, sub = out
            self.dump[f"{self.prefix}.L{i}.sub"] = sub.feats
        self.coords(f"{self.prefix}.L{i + 1}.coords", h.coords)
        self.feats(f"{self.prefix}.L{i + 1}.in", h.feats)

    def head(self, module, args, out):
        if self.prefix is None:
            return
        self.dump[f"{self.prefix}.out"] = out.feats


def build_pipeline():
    """Upstream's `from_pretrained`, with the background remover stubbed.

    The input carries alpha, so `preprocess_image` never calls it, but it is
    constructed at load, and constructing BiRefNet imports kornia, which imports
    `flash_attn` at module level. Same stub as CrawCity's `gen_trellis2.py`.
    """
    bootstrap()
    patch_dinov3()
    import trellis2.pipelines.rembg as rembg

    class NoRembg:
        def __init__(self, *a, **k):
            self.model = None

        def to(self, *a, **k):
            return self

        cuda = cpu = to

        def __call__(self, image):
            raise RuntimeError("background removal is stubbed; feed an RGBA cutout")

    rembg.BiRefNet = NoRembg
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    pipe = Trellis2ImageTo3DPipeline.from_pretrained(REPO)
    pipe.cuda()
    return pipe


def install(pipe, dump, level_rows):
    """Hooks on every model the run touches. Returns the flow and decoder recorders."""
    calls = FlowCalls(dump)
    for k, m in pipe.models.items():
        if "flow_model" in k:
            m.register_forward_hook(calls, with_kwargs=True)
    decoders = {k: DecoderLevels(dump, pipe.models[k], level_rows)
                for k in ("shape_slat_decoder", "tex_slat_decoder")}

    # DINOv3's input after upstream's resize and ImageNet normalisation — host
    # work the `trellis2-cond` graph deliberately leaves out.
    dino = {"res": None}

    def dino_input(module, args):
        dump[f"dino{dino['res']}.image"] = args[0]
    pipe.image_cond_model.model.embeddings.register_forward_pre_hook(dino_input)
    return calls, decoders, dino


def run_sampler(dump, calls, stage, sampler, model, noise, cond, params, coords=None,
                **kwargs):
    """`sampler.sample(...)`, keeping what it computes per step.

    `ret.pred_x_t` and `ret.pred_x_0` are upstream's own per-step lists. The
    post-guidance velocity is not kept by upstream, so `_get_model_prediction` is
    wrapped on this sampler instance for the duration; it returns what it
    returned. The `t` it sees is the Python float the guidance interval is
    tested against, which is also recorded.
    """
    vs, ts = [], []
    orig = sampler._get_model_prediction

    def recorded(model, x_t, t, cond=None, **kw):
        x0, eps, v = orig(model, x_t, t, cond, **kw)
        vs.append(feats(v).detach().cpu())
        ts.append(t)
        return x0, eps, v

    sampler._get_model_prediction = recorded
    calls.begin(stage, cond["cond"], coords)
    try:
        ret = sampler.sample(model, noise, **cond, **params, **kwargs, verbose=True,
                             tqdm_desc=stage)
    finally:
        del sampler._get_model_prediction
        calls.end()
    dump[f"{stage}.noise"] = feats(noise)
    dump[f"{stage}.steps.x"] = torch.stack([feats(x).detach().cpu() for x in ret.pred_x_t])
    dump[f"{stage}.steps.x0"] = torch.stack([feats(x).detach().cpu() for x in ret.pred_x_0])
    dump[f"{stage}.steps.v"] = torch.stack(vs)
    dump[f"{stage}.t"] = ts
    dump[f"{stage}.params"] = {**params, "sigma_min": sampler.sigma_min}
    return ret


# --- `Trellis2ImageTo3DPipeline`'s stage methods, transcribed ------------------

def get_cond(pipe, dump, dino, image, resolution, include_neg_cond=True):
    pipe.image_cond_model.image_size = resolution
    if pipe.low_vram:
        pipe.image_cond_model.to(pipe.device)
    dino["res"] = resolution
    cond = pipe.image_cond_model(image)
    dump[f"cond{resolution}"] = cond
    if pipe.low_vram:
        pipe.image_cond_model.cpu()
    if not include_neg_cond:
        return {'cond': cond}
    neg_cond = torch.zeros_like(cond)
    return {
        'cond': cond,
        'neg_cond': neg_cond,
    }


def sample_sparse_structure(pipe, dump, calls, cond, resolution, num_samples=1,
                            sampler_params={}):
    flow_model = pipe.models['sparse_structure_flow_model']
    reso = flow_model.resolution
    in_channels = flow_model.in_channels
    noise = torch.randn(num_samples, in_channels, reso, reso, reso).to(pipe.device)
    sampler_params = {**pipe.sparse_structure_sampler_params, **sampler_params}
    if pipe.low_vram:
        flow_model.to(pipe.device)
    z_s = run_sampler(dump, calls, "ss", pipe.sparse_structure_sampler, flow_model,
                      noise, cond, sampler_params).samples
    if pipe.low_vram:
        flow_model.cpu()

    decoder = pipe.models['sparse_structure_decoder']
    if pipe.low_vram:
        decoder.to(pipe.device)
    logits = decoder(z_s)
    dump["ss.logits"] = logits
    decoded = logits > 0
    if pipe.low_vram:
        decoder.cpu()
    if resolution != decoded.shape[2]:
        ratio = decoded.shape[2] // resolution
        decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
    dump["ss.occupancy"] = decoded
    coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()
    dump["ss.coords"] = coords
    return coords


def normalization(pipe, which, device):
    n = getattr(pipe, f"{which}_slat_normalization")
    std = torch.tensor(n['std'])[None].to(device)
    mean = torch.tensor(n['mean'])[None].to(device)
    return std, mean


def sample_shape_slat(pipe, dump, calls, stage, cond, flow_model, coords, sampler_params={}):
    from trellis2.modules.sparse import SparseTensor
    noise = SparseTensor(
        feats=torch.randn(coords.shape[0], flow_model.in_channels).to(pipe.device),
        coords=coords,
    )
    sampler_params = {**pipe.shape_slat_sampler_params, **sampler_params}
    if pipe.low_vram:
        flow_model.to(pipe.device)
    slat = run_sampler(dump, calls, stage, pipe.shape_slat_sampler, flow_model, noise,
                       cond, sampler_params, coords=coords).samples
    if pipe.low_vram:
        flow_model.cpu()
    dump[f"{stage}.coords"] = coords
    dump[f"{stage}.samples"] = slat.feats

    std, mean = normalization(pipe, "shape", slat.device)
    slat = slat * std + mean
    dump[f"{stage}.slat"] = slat.feats
    return slat


def sample_shape_slat_cascade(pipe, dump, calls, decoders, lr_cond, cond, flow_model_lr,
                              flow_model, lr_resolution, resolution, coords,
                              sampler_params={}, max_num_tokens=49152):
    # LR
    slat = sample_shape_slat(pipe, dump, calls, f"shape{lr_resolution}", lr_cond,
                             flow_model_lr, coords, sampler_params)

    # Upsample
    dec = pipe.models['shape_slat_decoder']
    if pipe.low_vram:
        dec.to(pipe.device)
        dec.low_vram = True
    decoders['shape_slat_decoder'].prefix = "up"
    hr_coords = dec.upsample(slat, upsample_times=4)
    decoders['shape_slat_decoder'].prefix = None
    if pipe.low_vram:
        dec.cpu()
        dec.low_vram = False
    dump["up.coords"] = hr_coords
    hr_resolution = resolution
    while True:
        quant_coords = torch.cat([
            hr_coords[:, :1],
            ((hr_coords[:, 1:] + 0.5) / lr_resolution * (hr_resolution // 16)).int(),
        ], dim=1)
        coords = quant_coords.unique(dim=0)
        num_tokens = coords.shape[0]
        if num_tokens < max_num_tokens or hr_resolution == 1024:
            if hr_resolution != resolution:
                print(f"Due to the limited number of tokens, the resolution is reduced to {hr_resolution}.")
            break
        hr_resolution -= 128
    dump["up.quant_coords"] = coords
    dump["hr_resolution"] = hr_resolution

    slat = sample_shape_slat(pipe, dump, calls, f"shape{resolution}", cond, flow_model,
                             coords, sampler_params)
    return slat, hr_resolution


def sample_tex_slat(pipe, dump, calls, stage, cond, flow_model, shape_slat, sampler_params={}):
    import torch.nn as nn
    std, mean = normalization(pipe, "shape", shape_slat.device)
    shape_slat = (shape_slat - mean) / std

    in_channels = flow_model.in_channels if isinstance(flow_model, nn.Module) else flow_model[0].in_channels
    noise = shape_slat.replace(feats=torch.randn(shape_slat.coords.shape[0], in_channels - shape_slat.feats.shape[1]).to(pipe.device))
    sampler_params = {**pipe.tex_slat_sampler_params, **sampler_params}
    if pipe.low_vram:
        flow_model.to(pipe.device)
    slat = run_sampler(dump, calls, stage, pipe.tex_slat_sampler, flow_model, noise, cond,
                       sampler_params, coords=shape_slat.coords,
                       concat_cond=shape_slat).samples
    if pipe.low_vram:
        flow_model.cpu()
    dump[f"{stage}.coords"] = shape_slat.coords
    dump[f"{stage}.samples"] = slat.feats

    std, mean = normalization(pipe, "tex", slat.device)
    slat = slat * std + mean
    dump[f"{stage}.slat"] = slat.feats
    return slat


def decode_latent(pipe, dump, decoders, shape_slat, tex_slat, resolution):
    dec = pipe.models['shape_slat_decoder']
    dec.set_resolution(resolution)
    if pipe.low_vram:
        dec.to(pipe.device)
        dec.low_vram = True
    decoders['shape_slat_decoder'].prefix = "shapedec"
    meshes, subs = dec(shape_slat, return_subs=True)
    decoders['shape_slat_decoder'].prefix = None
    if pipe.low_vram:
        dec.cpu()
        dec.low_vram = False

    tdec = pipe.models['tex_slat_decoder']
    if pipe.low_vram:
        tdec.to(pipe.device)
    decoders['tex_slat_decoder'].prefix = "texdec"
    tex_voxels = tdec(tex_slat, guide_subs=subs) * 0.5 + 0.5
    decoders['tex_slat_decoder'].prefix = None
    if pipe.low_vram:
        tdec.cpu()

    assert len(meshes) == 1 and len(tex_voxels) == 1, "one sample per run"
    m, v = meshes[0], tex_voxels[0]
    dump["mesh.vertices"] = m.vertices
    dump["mesh.faces"] = m.faces.int()
    m.fill_holes()
    dump["mesh_filled.vertices"] = m.vertices
    dump["mesh_filled.faces"] = m.faces.int()
    # `texdec.out * 0.5 + 0.5` on the shape decoder's last coords; both of those
    # are already stored.
    assert torch.equal(v.coords.cpu(), dump.tensors["shapedec.L4.coords"])
    dump["resolution"] = resolution
    print(f"  mesh {m.vertices.shape[0]} vertices, {m.faces.shape[0]} faces; "
          f"{v.feats.shape[0]} texture voxels")


@torch.no_grad()
def run(pipe, dump, image, *, seed, pipeline_type, max_num_tokens=49152, level_rows=None):
    """`Trellis2ImageTo3DPipeline.run`, transcribed, with saves."""
    calls, decoders, dino = install(pipe, dump, level_rows)

    dump["image_rgba"] = torch.from_numpy(np.asarray(image.convert("RGBA")).copy())
    image = pipe.preprocess_image(image)
    dump["image_pre"] = torch.from_numpy(np.asarray(image).copy())
    torch.manual_seed(seed)
    cond_512 = get_cond(pipe, dump, dino, [image], 512)
    cond_1024 = get_cond(pipe, dump, dino, [image], 1024) if pipeline_type != '512' else None
    ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}[pipeline_type]
    coords = sample_sparse_structure(pipe, dump, calls, cond_512, ss_res)
    print(f"  {coords.shape[0]} voxels at {ss_res}^3")
    if pipeline_type == '512':
        shape_slat = sample_shape_slat(pipe, dump, calls, "shape512", cond_512,
                                       pipe.models['shape_slat_flow_model_512'], coords)
        tex_slat = sample_tex_slat(pipe, dump, calls, "tex512", cond_512,
                                   pipe.models['tex_slat_flow_model_512'], shape_slat)
        res = 512
    elif pipeline_type == '1024_cascade':
        shape_slat, res = sample_shape_slat_cascade(
            pipe, dump, calls, decoders, cond_512, cond_1024,
            pipe.models['shape_slat_flow_model_512'], pipe.models['shape_slat_flow_model_1024'],
            512, 1024, coords, max_num_tokens=max_num_tokens)
        tex_slat = sample_tex_slat(pipe, dump, calls, "tex1024", cond_1024,
                                   pipe.models['tex_slat_flow_model_1024'], shape_slat)
    else:
        raise SystemExit(f"pipeline type {pipeline_type} is not transcribed here")
    torch.cuda.empty_cache()
    decode_latent(pipe, dump, decoders, shape_slat, tex_slat, res)

    dump["seed"] = seed
    dump["pipeline_type"] = pipeline_type
    dump["max_num_tokens"] = max_num_tokens
    dump["shape_slat_normalization"] = pipe.shape_slat_normalization
    dump["tex_slat_normalization"] = pipe.tex_slat_normalization
    return dump


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", type=Path,
                   default=CHECKOUT / "assets" / "example_image" / "T.png")
    p.add_argument("--pipeline", default="1024_cascade", choices=["512", "1024_cascade"])
    p.add_argument("--seed", type=int, default=42, help="upstream's default")
    p.add_argument("--out", type=Path, default=None,
                   help="default gen/graphs/trellis2-pipe-<pipeline>")
    p.add_argument("--flow-dtype", default="bf16", choices=["bf16", "fp16"],
                   help="the flow models' torso dtype. bf16 is upstream's; fp16 is "
                        "what tools/export_trellis2.py exports, and the reference a "
                        "free-running port is compared against: guidance at 7.5 "
                        "turns the bf16/fp16 rounding difference into different "
                        "voxels (29 of 3537 at 512), identically in torch and Lava")
    p.add_argument("--level-rows", type=int, default=None,
                   help="store at most this many feature rows per decoder level; "
                        "coords are always complete")
    a = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA; flex_gemm and the sparse attention are CUDA-only")
    # As the exporters do: TF32's 10-bit mantissa reads as ~2e-4 relative error,
    # the same order as a real bug.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    image_path = a.image.resolve()
    if not image_path.is_file():
        raise SystemExit(f"no image at {image_path}")
    suffix = "" if a.flow_dtype == "bf16" else f"-{a.flow_dtype}"
    out = (a.out or GEN / "graphs" / f"trellis2-pipe-{a.pipeline}{suffix}").resolve()
    print(f"trellis2 pipeline {a.pipeline}: {image_path.name}, seed {a.seed}, "
          f"flow models {a.flow_dtype}")
    pipe = build_pipeline()          # chdirs into the checkout
    if a.flow_dtype == "fp16":
        # `convert_to`, as the exporter does: only the torso, and the rope
        # table stays complex.
        for k, m in pipe.models.items():
            if "flow_model" in k:
                m.convert_to(torch.float16)
    from PIL import Image
    dump = run(pipe, Dump(), Image.open(image_path), seed=a.seed, pipeline_type=a.pipeline,
               level_rows=a.level_rows)
    dump["image"] = str(image_path)
    dump["flow_dtype"] = a.flow_dtype
    dump.write(out)
    print("peak GPU:", round(torch.cuda.max_memory_allocated() / 2**30, 2), "GB")
