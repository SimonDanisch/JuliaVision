# The int8 tensor cores, and what it would take to use them

Measured on a Radeon 8060S (RDNA 3.5, RADV), 2026-09-20, while taking a
Qwen-Image 2.1 denoising step from 12.5 s to 7.05 s.

## The number that matters

With staging removed from the k-loop — stage once, then run the same tiles
through the cooperative-matrix units — the two component types measure:

| | MAC-only rate |
| --- | --- |
| `Int8 -> Int32` cooperative matrices | **74.7 TOP/s** |
| `Float16 -> Float32` cooperative matrices | **24.4 TFLOP/s** |

The int8 units are **3.1x** the fp16 ones. That is the whole reason this is
worth writing down: the denoiser spends 109 ms a layer in products, half the
step, and those products are int8 weights in the checkpoint already.

## Where the current kernel sits

`q8gemm` stages the packed int8 weight, dequantises it into fp16 tiles and
multiplies on the fp16 units. At Qwen-Image's shapes it reaches **20.8 TOP/s**,
which is 85% of the 24.4 fp16 ceiling. There is nothing left in that direction:
the kernel is close to what fp16 can do on this device.

## What a W8A8 kernel gets today

Three prototypes, same shapes (`12288 x 4096 x 4224` and friends), all exact
against an integer reference:

| | rate |
| --- | --- |
| byte-at-a-time staging, 64x64 tile | 15.4 TOP/s |
| byte-at-a-time staging, 128x64 tile | 15.6 TOP/s |
| packed 32-bit global loads, byte LDS stores, 128x64 | 17.3 TOP/s |
| 4-wide int8 vector LDS, 128x64 | 19.8 TOP/s |
| 4-wide int8 vector LDS, 128x128 | **22.4 TOP/s** |

So a straightforward int8 kernel lands at parity with the fp16-tile one — 30%
of the int8 ceiling — and every improvement so far has been to STAGING, not to
the arithmetic. That is the shape of the problem: fp16 math is slow enough to
hide the staging behind it, and int8 math at 3x is not.

Getting from 22 to something near 74 needs what `q8gemm` and `Mantle`'s fp16
GEMM already have and these prototypes do not:

* double buffering, so a block's staging overlaps the previous block's math
  (both barriers per k-block are currently exposed),
* wider staging per thread, and a k-block deep enough to amortise the two
  barriers,
* a tile sweep of the kind `q8gemm_tile` records, at the shapes that matter.

## What is already in place

* **Lava emits signed 8-bit cooperative matrices** (`dev/Lava`, "Signed 8-bit
  cooperative matrices"). The component type is signless, like every other
  integer the emitter makes, and `OpCooperativeMatrixMulAddKHR` carries the
  signedness in its operands word. A 16x16x16 int8 product is exact, and so is
  a 12288x4096x4224 one. Before that fix an int8 matrix over a shared array
  failed SPIR-V validation.
* **The weights are already packed the way such a kernel wants them.**
  `convrotqint8` stores four output rows to a `UInt32` word, which is exactly a
  4-wide int8 vector for the A tile.

## What is not

* **Activation quantisation.** W8A8 means the activation is int8 too, with a
  per-token scale, which is what Comfy's checkpoint expects — ConvRot exists to
  make that accurate. Nothing in DNNKernels quantises an activation yet: it
  needs a pass after the ConvRot transform and a scale that reaches the GEMM's
  epilogue, where the product becomes `acc * w_scale[m] * a_scale[n]`.
* **The accuracy question.** Every number above is exactness against an integer
  reference, which says the kernel computes the product it claims. Whether the
  MODEL survives int8 activations is a separate measurement against the fp16
  activation path and, past that, against the reference implementation.

The prototypes are in `tmp/w8a8_probe.jl` on the machine they were measured on;
they are scratch, not a package path, and the numbers above are what they were
for.
