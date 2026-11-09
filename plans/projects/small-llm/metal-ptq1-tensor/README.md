# PTQ1 matrix product on Metal tensor units (unfinished)

`DNNKernelsMetalExt.jl` multiplies a tiled PTQ1 matrix (Bonsai's ternary weights,
`DNNKernels/src/ptq1.jl`) by Float16 columns with Metal 4 tensor operations on the M5:
each workgroup of 128 threads decodes 64-row tiles into threadgroup memory and
accumulates with `TensorOpsMatmul2D` into cooperative tensors.

It is kept here, outside the package, because it cannot load yet:

- It imports `Metal.MtlCooperativeTensor`, which no Metal.jl defines. Metal's archived
  prototype of that API (`Metal/docs/archive/mac-20261101`) failed to compile on the M5.
- It adds methods to `DNNKernels.ptq1_tensor_available` and `DNNKernels.ptq1_mm_tensor!`,
  which DNNKernels does not define. `ptq1mul!` would have to call them when the device
  has tensor units.
- `DNNKernels/Project.toml` has no Metal weak dependency or extension entry for it.

It was never run.
