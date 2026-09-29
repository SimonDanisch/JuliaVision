"""Run upstream's `o_voxel.postprocess.to_glb` on a pipeline dump and record its steps.

    ~/tools/TRELLIS.2/.venv/bin/python tools/dump_trellis2_glb.py                  # 1024_cascade
    ~/tools/TRELLIS.2/.venv/bin/python tools/dump_trellis2_glb.py --pipeline 512

`dump_trellis2_pipeline.py` stops at the extracted mesh (`mesh_filled.*`). What
`to_glb` does after it (a second `fill_holes`, the narrow-band rebuild, the
simplification, the UV unwrap and the bake) runs in cumesh, nvdiffrast and cv2,
and this records each step's result from upstream's own `to_glb`, called with
the arguments `example.py` passes. Nothing is reimplemented: the steps are
observed by wrapping the cumesh methods and `cv2.inpaint` `to_glb` calls.

The inputs are the dump's, exactly what upstream's `decode_latent` hands
`to_glb`: `mesh_filled.*`, the texture voxels `texdec.out * 0.5 + 0.5` on
`shapedec.L4.coords`, `voxel_size = 1 / resolution`, the unit cube.

What is recorded (`glb-<decimation>.safetensors` beside the dump, the GLB in
`gen/trellis2-out`):

| key | what |
|---|---|
| `glb.filled.*` | the mesh after `to_glb`'s own `fill_holes` |
| `glb.remeshed.*` | `remesh_narrow_band_dc`'s output |
| `glb.simplified.*` | after `simplify(decimation_target)` |
| `glb.unwrapped.*` | `uv_unwrap`'s vertices, faces, uvs and vertex map |
| `glb.basecolor.raw`, `glb.basecolor`, `glb.mask` | the baked base colour before and after inpainting, and the covered texels |

cumesh's `simplify` resolves its collapses with atomics, so two runs need not give
the same mesh; everything before it is deterministic.

Upstream: https://github.com/microsoft/TRELLIS.2
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_trellis2 import GEN, bootstrap


def observe(dump):
    """Wrap the steps `to_glb` takes so each result lands in `dump`."""
    import cv2
    import cumesh
    import cumesh.remeshing

    def put(key, t):
        t = t.detach() if isinstance(t, torch.Tensor) else torch.as_tensor(t)
        assert key not in dump, f"{key} recorded twice"
        dump[key] = t.contiguous().cpu()

    fill_holes = cumesh.CuMesh.fill_holes

    def fill_holes_(self, *a, **k):
        out = fill_holes(self, *a, **k)
        v, f = self.read()
        put("glb.filled.vertices", v)
        put("glb.filled.faces", f.int())
        return out
    cumesh.CuMesh.fill_holes = fill_holes_

    remesh = cumesh.remeshing.remesh_narrow_band_dc

    def remesh_(*a, **k):
        v, f = remesh(*a, **k)
        put("glb.remeshed.vertices", v)
        put("glb.remeshed.faces", f.int())
        return v, f
    cumesh.remeshing.remesh_narrow_band_dc = remesh_

    simplify = cumesh.CuMesh.simplify

    def simplify_(self, *a, **k):
        out = simplify(self, *a, **k)
        v, f = self.read()
        put("glb.simplified.vertices", v)
        put("glb.simplified.faces", f.int())
        return out
    cumesh.CuMesh.simplify = simplify_

    uv_unwrap = cumesh.CuMesh.uv_unwrap

    def uv_unwrap_(self, *a, **k):
        out = uv_unwrap(self, *a, **k)
        for name, t in zip(("vertices", "faces", "uvs", "vmaps"), out):
            put(f"glb.unwrapped.{name}", t.int() if name in ("faces", "vmaps") else t)
        return out
    cumesh.CuMesh.uv_unwrap = uv_unwrap_

    # The first `inpaint` is the base colour's (radius 3); the others are the
    # one-channel textures.
    inpaint = cv2.inpaint
    seen = {"first": True}

    def inpaint_(img, mask, radius, flags):
        out = inpaint(img, mask, radius, flags)
        if seen["first"]:
            seen["first"] = False
            put("glb.basecolor.raw", torch.from_numpy(np.ascontiguousarray(img)))
            put("glb.mask", torch.from_numpy(np.ascontiguousarray(mask == 0).astype(np.uint8)))
            put("glb.basecolor", torch.from_numpy(np.ascontiguousarray(out)))
        return out
    cv2.inpaint = inpaint_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", default="1024_cascade")
    ap.add_argument("--decimation", type=int, default=1_000_000)   # example.py's
    ap.add_argument("--texture", type=int, default=4096)
    args = ap.parse_args()

    src = GEN / "graphs" / f"trellis2-pipe-{args.pipeline}"
    keys = ("mesh_filled.vertices", "mesh_filled.faces", "shapedec.L4.coords", "texdec.out")
    with safe_open(str(src / "pipeline.safetensors"), framework="pt") as f:
        d = {k: f.get_tensor(k) for k in keys}
    resolution = 1024 if args.pipeline == "1024_cascade" else 512

    bootstrap()
    import o_voxel.postprocess

    dump = {}
    observe(dump)
    coords = d["shapedec.L4.coords"][:, 1:].cuda()
    attrs = (d["texdec.out"].float() * 0.5 + 0.5).cuda()
    glb = o_voxel.postprocess.to_glb(
        vertices=d["mesh_filled.vertices"].cuda(),
        faces=d["mesh_filled.faces"].cuda(),
        attr_volume=attrs,
        coords=coords,
        attr_layout={'base_color': slice(0, 3), 'metallic': slice(3, 4),
                     'roughness': slice(4, 5), 'alpha': slice(5, 6)},
        voxel_size=1 / resolution,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=args.decimation,
        texture_size=args.texture,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
    )
    out = GEN / "trellis2-out" / f"upstream-{args.pipeline}-{args.decimation}.glb"
    out.parent.mkdir(parents=True, exist_ok=True)
    glb.export(str(out))
    save_file(dump, str(src / f"glb-{args.decimation}.safetensors"))
    for k, v in sorted(dump.items()):
        print(f"  {k:<26} {str(tuple(v.shape)):<22} {str(v.dtype).removeprefix('torch.')}")
    print(f"-> {out}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
