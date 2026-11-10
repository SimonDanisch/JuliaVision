# Trellis2Runner

[TRELLIS.2](https://github.com/microsoft/TRELLIS.2) image to textured 3D on
Lava: DINOv3 conditioning, a dense flow over a 16³ latent decoded to 64³
occupancy, sparse shape and texture flows, two sparse decoders, then cumesh's
mesh post-processing (dual grid, hole filling, narrow-band remesh, simplify),
UV charting and a PBR texture bake. The `512` and `1024_cascade` variants.

`generate(p, rgba; noise, texture, decimation)` takes a `(4, W, H)` `UInt8`
picture and returns a mesh with one PBR material (base colour and
metallic-roughness). The stages are public too: `sparsestructure`, `shapeslat`,
`texslat`, `decode`, `remesh`, `fillholes`, `simplify`, `texturedmesh`.

## Status

Ported, but **not packaged**: its graphs and decoder weights are not published
as artifacts yet, so it cannot be used from a clean checkout. It has no tests
and is not in the workspace environment. Publishing the artifacts is the
remaining step.
