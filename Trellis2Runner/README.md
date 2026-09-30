# Trellis2Runner

[TRELLIS.2](https://github.com/microsoft/TRELLIS.2) image to textured 3D on
Lava: DINOv3 conditioning, a dense flow over a 16³ latent decoded to 64³
occupancy, sparse shape and texture flows, two sparse decoders, then cumesh's
mesh post-processing (dual grid, hole filling, narrow-band remesh, simplify),
UV charting and a PBR texture bake. The `512` and `1024_cascade` variants.

```julia
using Trellis2Runner, FileIO, MeshIO
import Mantle

p = Trellis2("gen/graphs", "ckpts"; backend = Mantle.defaultbackend(), cascade = true)
mesh = generate(p, rgba; noise = Noise(42), texture = 4096, decimation = 500_000)
save("model.glb", mesh)      # one PBR material: base colour and metallic-roughness
```

`rgba` is `(4, W, H)` `UInt8`. The stages are public too: `sparsestructure`,
`shapeslat`, `texslat`, `decode`, `remesh`, `fillholes`, `simplify`,
`texturedmesh`.

## Status

Ported, and **not runnable from a clean checkout**. Unlike every other runner it
reads explicit paths rather than artifacts, and nothing is published yet:

  * the graphs come from `tools/export_trellis2.py --part {ss, ssdec, cond,
    shape512, shape1024, tex512, tex1024, sampling}`, run with a TRELLIS.2
    checkout's Python environment;
  * the two sparse decoders read upstream's
    `shape_dec_next_dc_f16c32_fp16.safetensors` and
    `tex_dec_next_dc_f16c32_fp16.safetensors` directly.

It has no tests and is not in the workspace environment. Binding the exports as
artifacts is what would change both.
