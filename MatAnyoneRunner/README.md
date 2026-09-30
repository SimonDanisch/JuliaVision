# MatAnyoneRunner

[MatAnyone2](https://github.com/pq-yang/MatAnyone) video matting on Lava: given a
mask of the subject on one frame, a soft alpha matte for every frame. It carries
a mask through a clip; finding the subject is SAM 2's job. **Non-commercial**
(S-Lab License 1.0).

## Matting a clip

```julia
using MatAnyoneRunner, SAM2Runner

seed  = segment(frames[1], [(0.68, 0.7), (0.675, 0.52)])   # SAM 2 marks the subject
prop  = matanyonepropagator()
alpha = prop(frames, Dict(1 => seed))    # (W, H, n) UInt8, 0..255
```

`frames` is a vector of `(W, H)` colour matrices; sizes are padded to multiples of
16 internally. `seeds` maps frame indices to `UInt8` masks, anything above 127
counting as the subject; frames before the first seed stay zero.
`progress = (k, n) -> …` reports as it goes.

`matanyonemodel()` and `runmatanyone(model, image, mask)` are the single-frame
level underneath. There the mask is Float32 in **0..255**; a 0/1 mask silently
gives an all-zero matte.

## Examples

SAM 2 marks the mechanic in the first frame of a push-in, 30% closer and 4
degrees turned by the end, and MatAnyone carries him through the other 23:
[`docs/examples/matanyone.jl`](../docs/examples/matanyone.jl).

<img src="../media/matanyone/pushin.jpg" width="960">

Frames 1, 12 and 24, and their mattes. **113 ms** a frame at 640x368 on a
Radeon 8060S (RADV, 2026-09-30).

## Assets

| artifact | size | |
|---|---|---|
| `matanyone` | 136 MB | graphs and weights |
| `matanyone-refs` | 1.2 GB | PyTorch activations for the parity test only |
