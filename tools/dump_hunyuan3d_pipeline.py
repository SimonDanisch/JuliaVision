"""Run the real Hunyuan3D-2.1 shape pipeline and dump every intermediate.

    uv run tools/dump_hunyuan3d_pipeline.py --octree 64      # bring-up
    uv run tools/dump_hunyuan3d_pipeline.py                  # the real 384 grid

`export_hunyuan3d.py` writes four graphs and a reference for each, and every one
of them matches on Lava. What is *not* covered by those references is the glue
between them, which is where a port of a diffusion model actually goes wrong: the
image preprocessing, the sigma schedule, the classifier-free-guidance
combination, the order of the two rows in the batch, the `1/scale_factor` before
the VAE, the chunking of the query grid and the marching-cubes conventions. None
of that is in a graph. All of it is in this file's output.

**This is a transcription of `Hunyuan3DDiTFlowMatchingPipeline.__call__`, not a
reimplementation of it.** The body below is upstream's, line for line, with
`_save` calls interleaved; the pipeline object it drives is upstream's own,
built by upstream's `from_single_file`. Anything that reads like a simplification
is a bug. The reason for transcribing at all rather than passing a callback is
that `__call__`'s callback fires once per step with the scheduler output and
cannot see `cond`, the preprocessed image, the per-step model output before
guidance, or anything downstream of the loop.

Everything lands in one safetensors file plus a JSON of the scalars. The Julia
side reads `latents_init` from it rather than seeding its own RNG: matching
PyTorch's Philox bit for bit is not what this port is about, and a pipeline that
starts from the same noise is the only way a per-step comparison means anything.

**Grid resolution is the one knob that matters for turnaround.** At the
pipeline's default `octree_resolution=384` the grid is 385^3 = 57.07M query
points, 7134 chunks of 8000, and about a minute of GPU time; the dumped field is
114 MB at fp16. `--octree 64` is 275k points and 35 chunks, which is the same
code path in seconds. Bring the Julia side up against 64, then confirm at 384.

Upstream: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
License: TENCENT HUNYUAN NON-COMMERCIAL.
"""

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

# Paths only — importing this module does not install the `_shell` stubs, which
# matters because this script needs the real package `__init__`s to run.
from export_hunyuan3d import CHECKOUT, CKPT_DIR, GEN


def bootstrap():
    """Import upstream for real, skipping exactly one `__init__`.

    `export_hunyuan3d.py` replaces four packages with bare module objects because
    the denoiser is a plain `nn.Module` and none of the `__init__` bodies are on
    its path. Here they are: `pipelines.py` does `from .models.autoencoders import
    ShapeVAE`, and a shell has a search path and no attributes, so every
    subpackage has to be real.

    The single exception is the top-level `hy3dshape/__init__.py`, which imports
    `postprocessors` and through it `pymeshlab`. That is mesh cleanup —
    `FaceReducer`, `FloaterRemover`, `MeshSimplifier` — applied after this
    pipeline returns, and it is a 200 MB dependency for nothing the port touches.
    Skipping that one file leaves every module this script imports genuine.
    """
    pkg = CHECKOUT / "hy3dshape"
    if not pkg.is_dir():
        raise SystemExit(
            f"no hy3dshape package at {pkg}\n"
            "  git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1")
    if str(CHECKOUT) not in sys.path:
        sys.path.insert(0, str(CHECKOUT))
    if "hy3dshape" not in sys.modules:
        m = types.ModuleType("hy3dshape")
        m.__path__ = [str(pkg)]
        m.__package__ = "hy3dshape"
        sys.modules["hy3dshape"] = m
    return pkg


class Dump:
    """Tensors and scalars for one run, written once at the end.

    Saved on the CPU and in the dtype they were computed in — an fp16 activation
    widened to fp32 on the way out is a reference for a run that never happened.
    The exception is the grid field, which upstream itself computes as `.float()`
    and which is 228 MB at fp32; that one is narrowed on the way out and says so.
    """

    def __init__(self):
        self.tensors = {}
        self.meta = {}

    def __setitem__(self, name, value):
        if isinstance(value, torch.Tensor):
            self.tensors[name] = value.detach().contiguous().cpu()
        else:
            self.meta[name] = value

    def write(self, out: Path):
        out.mkdir(parents=True, exist_ok=True)
        save_file(self.tensors, str(out / "pipeline.safetensors"))
        (out / "pipeline.json").write_text(json.dumps(self.meta, indent=1))
        total = sum(t.numel() * t.element_size() for t in self.tensors.values())
        print(f"\n{len(self.tensors)} tensors, {total / 1e6:.1f} MB -> {out}")
        for k, v in self.tensors.items():
            print(f"  {k:<16} {str(tuple(v.shape)):<22} {str(v.dtype).removeprefix('torch.')}")


def build_pipeline(device: str, dtype):
    bootstrap()
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    ckpt = CKPT_DIR / "model.fp16.ckpt"
    if not ckpt.is_file():
        raise SystemExit(
            f"no {ckpt} — accept the licence, then "
            "`huggingface-cli download tencent/Hunyuan3D-2.1`")
    return Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
        str(ckpt), str(CKPT_DIR / "config.yaml"), device=device, dtype=dtype)


def run(pipe, dump, image_path, *, steps, guidance_scale, seed, box_v,
        octree_resolution, mc_level, num_chunks):
    """`Hunyuan3DDiTFlowMatchingPipeline.__call__`, transcribed, with saves."""
    from PIL import Image

    device, dtype = pipe.device, pipe.dtype
    do_classifier_free_guidance = guidance_scale >= 0 and not (
        hasattr(pipe.model, "guidance_embed") and pipe.model.guidance_embed is True)
    if not do_classifier_free_guidance:
        raise SystemExit("this checkpoint was expected to use CFG; it does not")

    # --- conditioning ---------------------------------------------------------
    # `prepare_image` runs `ImageProcessorV2`: alpha-matte recentre with a 0.15
    # border, composite onto white, cubic resize to 512, then `/255*2-1`. The
    # Julia side has to reproduce this exactly, so both its inputs and its output
    # are dumped.
    image = Image.open(image_path)
    dump["image_rgba"] = torch.from_numpy(np.asarray(image.convert("RGBA")).copy())
    cond_inputs = pipe.prepare_image(image)
    image_t = cond_inputs.pop("image")
    dump["image_512"] = image_t.float()
    dump["mask_512"] = cond_inputs["mask"].float()

    # The conditioner's own `transform` — rescale (-1,1)->(0,1), bilinear resize
    # to 518 with antialias, centre crop, ImageNet normalise — is host work that
    # `hunyuan3d_cond` deliberately does not contain. Capturing its output is the
    # only way to tell a wrong resize apart from a wrong encoder.
    enc = pipe.conditioner.main_image_encoder
    with torch.no_grad():
        scaled = (image_t.to(device=device, dtype=dtype) - (-1)) / 2
        dump["image_518"] = enc.transform(scaled)

    cond = pipe.encode_cond(image=image_t, additional_cond_inputs=cond_inputs,
                            do_classifier_free_guidance=True, dual_guidance=False)
    # Row 0 is the image condition, row 1 is `unconditional_embedding`, which is
    # exact zeros rather than a learned null. Asserted, not assumed: the whole
    # guidance combination has its sign flipped if these two are swapped.
    assert cond["main"].shape[0] == 2
    assert cond["main"][1].abs().max().item() == 0.0, "row 1 is not the null embedding"
    dump["cond"] = cond["main"]

    batch_size = image_t.shape[0]

    # --- schedule -------------------------------------------------------------
    # NOTE (upstream's): this is slightly different from common usage, we start
    # from 0. `sigmas` runs 0 -> 1, `set_timesteps` appends a trailing 1.0, and
    # the final step therefore has `sigma_next - sigma == 0` and moves nothing.
    sigmas = np.linspace(0, 1, steps)
    from hy3dshape.pipelines import retrieve_timesteps
    timesteps, steps = retrieve_timesteps(pipe.scheduler, steps, device, sigmas=sigmas)
    dump["sigmas"] = pipe.scheduler.sigmas.float()
    dump["timesteps"] = timesteps.float()

    generator = torch.Generator(device=device).manual_seed(seed)
    latents = pipe.prepare_latents(batch_size, dtype, device, generator)
    dump["latents_init"] = latents

    # --- the loop -------------------------------------------------------------
    preds, traj = [], []
    for i, t in enumerate(timesteps):
        latent_model_input = torch.cat([latents] * 2)
        timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)
        timestep = timestep / pipe.scheduler.config.num_train_timesteps
        with torch.no_grad():
            noise_pred = pipe.model(latent_model_input, timestep, cond, guidance=None)
        preds.append(noise_pred.detach().cpu())

        noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
        traj.append(latents.detach().cpu())
        if i % 10 == 0 or i == len(timesteps) - 1:
            print(f"  step {i:>2}/{len(timesteps)}  t={t.item():8.3f}  "
                  f"|latents| {latents.float().abs().max().item():.4f}")

    # (steps, 2, 4096, 64) and (steps, 1, 4096, 64) — 52 and 26 MB at fp16. Worth
    # it: a mismatch that only appears at step 30 is a different bug from one at
    # step 0, and without these the Julia side cannot tell which it has.
    dump["noise_preds"] = torch.stack(preds)
    dump["latents_traj"] = torch.stack(traj)
    dump["latents_final"] = latents

    # --- decode ---------------------------------------------------------------
    # `_export`: divide by the VAE's scale factor, run `post_kl` + the decoder
    # transformer, then sweep the geometry decoder over a dense grid.
    with torch.no_grad():
        shape_latents = pipe.vae(1.0 / pipe.vae.scale_factor * latents)
    dump["shape_latents"] = shape_latents

    from hy3dshape.models.autoencoders.volume_decoders import generate_dense_grid_points
    bounds = [-box_v, -box_v, -box_v, box_v, box_v, box_v]
    xyz, grid_size, _ = generate_dense_grid_points(
        np.array(bounds[0:3]), np.array(bounds[3:6]), octree_resolution, indexing="ij")
    xyz = torch.from_numpy(xyz).to(device, dtype=dtype).contiguous().reshape(-1, 3)
    print(f"  grid {grid_size} = {xyz.shape[0]} points, "
          f"{-(-xyz.shape[0] // num_chunks)} chunks of {num_chunks}")

    # One full chunk, in and out, so the Julia side can check the sweep's
    # addressing without decoding a whole volume.
    dump["queries_0"] = xyz[:num_chunks].unsqueeze(0)
    with torch.no_grad():
        dump["logits_0"] = pipe.vae.geo_decoder(
            queries=xyz[:num_chunks].unsqueeze(0), latents=shape_latents)

    with torch.no_grad():
        grid_logits = pipe.vae.volume_decoder(
            shape_latents, pipe.vae.geo_decoder, bounds=box_v,
            num_chunks=num_chunks, octree_resolution=octree_resolution,
            enable_pbar=True)
    # Narrowed: upstream computes this as fp32 and it is 4 bytes x 57M at 384.
    # The comparison it feeds is against a field whose values live in about
    # (-30, 30), where fp16 has ~0.03 of resolution — far below the level-set
    # tolerance, and the mesh itself is dumped separately anyway.
    dump["grid_logits"] = grid_logits[0].half()
    dump["grid_stats"] = [grid_logits.min().item(), grid_logits.max().item(),
                          grid_logits.mean().item()]

    out = pipe.vae.surface_extractor(grid_logits, mc_level=mc_level, bounds=box_v,
                                     octree_resolution=octree_resolution)
    if out[0] is None:
        raise SystemExit("marching cubes produced nothing — see the traceback above")
    dump["mesh_v"] = torch.from_numpy(out[0].mesh_v)
    dump["mesh_f"] = torch.from_numpy(out[0].mesh_f.astype(np.int32))
    print(f"  mesh {out[0].mesh_v.shape[0]} vertices, {out[0].mesh_f.shape[0]} faces")

    dump["image"] = str(image_path)
    dump["steps"] = int(steps)
    dump["guidance_scale"] = guidance_scale
    dump["seed"] = seed
    dump["box_v"] = box_v
    dump["octree_resolution"] = octree_resolution
    dump["grid_size"] = [int(g) for g in grid_size]
    dump["mc_level"] = mc_level
    dump["num_chunks"] = num_chunks
    dump["scale_factor"] = pipe.vae.scale_factor
    dump["num_train_timesteps"] = pipe.scheduler.config.num_train_timesteps
    return dump


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", type=Path,
                   default=CHECKOUT.parent / "assets" / "demo.png")
    p.add_argument("--out", type=Path, default=GEN / "graphs" / "hunyuan3d-pipe")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--box-v", type=float, default=1.01)
    p.add_argument("--octree", type=int, default=384,
                   help="octree_resolution; the grid is (n+1)^3. 64 for bring-up")
    p.add_argument("--mc-level", type=float, default=0.0)
    p.add_argument("--num-chunks", type=int, default=8000,
                   help="must match the chunk `export_hunyuan3d.py --part geo` "
                        "was exported at, since that is a static shape")
    a = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA; the denoiser is 6.1 GB of fp16 and this is a "
                         "reference run, not an inspection")
    # As `export_hunyuan3d.py` does: TF32's 10-bit mantissa reads as ~2e-4
    # relative error, which is the same order as a real bug.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    if not a.image.is_file():
        raise SystemExit(f"no image at {a.image}")
    print(f"hunyuan3d pipeline: {a.image.name}, {a.steps} steps, "
          f"guidance {a.guidance_scale}, octree {a.octree}")
    pipe = build_pipeline("cuda", torch.float16)
    dump = run(pipe, Dump(), a.image, steps=a.steps, guidance_scale=a.guidance_scale,
               seed=a.seed, box_v=a.box_v, octree_resolution=a.octree,
               mc_level=a.mc_level, num_chunks=a.num_chunks)
    dump.write(a.out)
