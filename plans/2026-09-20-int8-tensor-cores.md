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

## What a W8A8 kernel gets, finished

`DNNKernels/src/w8a8.jl`, measured at `12288 x 4096 x 4224`:

| | rate |
| --- | --- |
| byte-at-a-time staging, 64x64 tile | 15.4 TOP/s |
| byte-at-a-time staging, 128x64 tile | 15.6 TOP/s |
| packed 32-bit global loads, byte LDS stores | 17.3 TOP/s |
| 4-wide int8 vector LDS | 19.8 TOP/s |
| 128x128 tile | 22.4 TOP/s |
| double buffered, int32 stored raw | 27.6 TOP/s |
| ... with the scale epilogue | 23.1 TOP/s |
| **A read straight from global, B staged, scale epilogue** | **28.6 TOP/s** |

1.38x the fp16-tile path, 38% of the int8 ceiling. Two things account for the
rest: the epilogue (27.6 -> 22.4 measured with the scales dropped, so it is the
per-component `getcomp`/`setcomp` and not the scales), and B's staging.

Two findings worth keeping:

* **A needs no staging and no repacking.** A weight tile is `(M, K)`
  column-major, which is exactly a `MatrixA` load; and `convrotqint8`'s four
  rows to a `UInt32` word ARE int8 `(M, K)` bytes. Staging A in shared memory
  instead costs 28.6 -> 23.1.
* **Guarding the double buffer is a wrong answer, not a slow one.** Writing
  `if kb + 1 < nb` around the prefetch and the store puts the two barriers at
  non-uniform control flow, and the result is correct for one, two and three k
  blocks and wrong from the fourth. Clamping the index instead is correct and
  faster.

## Why it is not on the path

Accuracy, not speed. Against an fp32 reference of the same dequantised weight
at `1024 x 4096 x 256`:

| activation | fp16 activation | int8 activation |
| --- | --- | --- |
| Gaussian | 0.036% rms | 0.87% rms |
| 1% outliers at 8 sigma | 0.036% rms | 3.5% rms |

12% of a step for 24x to 97x the error, on a path already more accurate than
the reference implementation. The kernel is in the tree with tests
(`test_w8a8.jl`) so the next person does not have to write it again to ask the
question; what would change the answer is a finer activation quantisation —
per-group scales along k, as the encoder's own W4A8 weights use — not a faster
kernel.

## What is already in place

* **Lava emits signed 8-bit cooperative matrices** (`dev/Lava`, "Signed 8-bit
  cooperative matrices"). The component type is signless, like every other
  integer the emitter makes, and `OpCooperativeMatrixMulAddKHR` carries the
  signedness in its operands word. Before that fix an int8 matrix over a shared
  array failed SPIR-V validation.
* **The quantiser**, `w8a8quantize`: a workgroup per activation column, its own
  scale, packed four to a word along k, padding columns quantised to zero.
