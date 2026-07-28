"""
Convolution: hand-written, type-parameterized, shared by every model.

ATen `convolution.default` in the reversed `(x, y, c, n)` layout. The weight is
a reversed `(C_out, C_in÷groups, KH, KW)`, i.e. indexed `(kx, ky, ci, co)`.

Kernel extent, stride, padding, dilation and groups are `Val` parameters so the
loop bounds and the group arithmetic fold at compile time. Only the plane
extents come from the launch geometry - resolution is never a type parameter.

Capability dispatch (scalar here, cooperative-matrix later) is selected at
instantiate from a device query, not here.
"""

"""
    accum(T)

Accumulator type for a reduction over `T`. Half precision accumulates in
Float32 because that is what the reference does - tensor cores take fp16
operands and accumulate fp32, and PyTorch's autocast relies on it. Summing a
few thousand products in fp16 instead drifts by percent, not ulps.
"""
@inline accum(::Type{Float16}) = Float32
@inline accum(::Type{T}) where {T} = T

@inline function conv2d(I, x, w, bias,
                        ::Val{SX}, ::Val{SY}, ::Val{PX}, ::Val{PY},
                        ::Val{DX}, ::Val{DY}, ::Val{GROUPS}) where {SX,SY,PX,PY,DX,DY,GROUPS}
    ox, oy, co, n = I
    @inbounds begin
        KX, KY, CIN = size(w, 1), size(w, 2), size(w, 3)
        A = accum(eltype(x))
        acc = bias === nothing ? zero(A) : A(bias[co])
        g = GROUPS == 1 ? 0 : (co - 1) ÷ (size(w, 4) ÷ GROUPS)
        bx = (ox - 1) * SX - PX
        by = (oy - 1) * SY - PY
        for ci in 1:CIN
            xc = g * CIN + ci
            for ky in 1:KY
                iy = by + (ky - 1) * DY + 1
                (iy < 1 || iy > size(x, 2)) && continue
                for kx in 1:KX
                    ix = bx + (kx - 1) * DX + 1
                    (ix < 1 || ix > size(x, 1)) && continue
                    acc = muladd(A(x[ix, iy, xc, n]), A(w[kx, ky, ci, co]), acc)
                end
            end
        end
        acc
    end
end

"""
    convolution!(out, x, w, bias, stride, padding, dilation, groups)

`stride`/`padding`/`dilation` are `(x, y)` in the reversed layout, i.e. the
reverse of the `(h, w)` pairs ATen carries.
"""
function convolution!(out, x, w, bias, stride, padding, dilation, groups;
                      ws=nothing, act::Symbol=:none)
    # Dense convolutions go through the implicit-GEMM kernel (conv_implicit.jl);
    # it is the only one with any data reuse. Grouped ones stay here — this model
    # has none, so that path is untuned.
    #
    # Same kernel on every backend, CPU included: the CPU run is what validates
    # the GPU run against the reference activations, so they have to be the same
    # source or the verification means nothing.
    s = (stride[1], stride[2])
    p = (padding[1], padding[2])
    d = (dilation[1], dilation[2])
    if groups == 1
        # Tensor cores when the extents land on the cooperative-matrix tile and
        # the operands are fp16 (i.e. the autocast export); the scalar
        # implicit-GEMM otherwise. Same arithmetic, ~30x apart on the layers
        # that dominate this model.
        conv_coopmat_applicable(out, x, w) &&
            return convolution_coopmat!(out, x, w, bias, s, p, d; ws, act)
        return convolution_igemm!(out, x, w, bias, s, p, d; act)
    end
    convolution_direct!(out, x, w, bias, stride, padding, dilation, groups)
    # The grouped path has no epilogue to fold into; this model has no grouped
    # convolutions, so it is left as a separate pass rather than duplicated.
    act === :relu && (out .= max.(out, zero(eltype(out))))
    out
end

"""One thread per output element, no reuse. Kept for grouped convolutions and as
the reference the implicit-GEMM kernel is checked against."""
function convolution_direct!(out, x, w, bias, stride, padding, dilation, groups)
    launch!(conv2d, out, x, w, bias,
            Val(stride[1]), Val(stride[2]), Val(padding[1]), Val(padding[2]),
            Val(dilation[1]), Val(dilation[2]), Val(groups))
end

"""
    convsize(inplane, k, stride, padding, dilation) -> Int

ATen's output extent, evaluated host-side so no kernel recomputes it.
"""
convsize(n, k, s, p, d) = (n + 2p - d * (k - 1) - 1) ÷ s + 1

"""
1-D convolution. `aten::convolution` covers 1/2/3-D under one schema and the
channel-attention blocks (channel_attn.py) use the 1-D form, so it is a
separate body rather than a 2-D one with a degenerate axis - the inner loop
should not carry a trip count of 1.

Layout `(x, c, n)`; weight reversed `(C_out, C_in÷groups, K)` = `(kx, ci, co)`.
"""
@inline function conv1d(I, x, w, bias, ::Val{SX}, ::Val{PX}, ::Val{DX},
                        ::Val{GROUPS}) where {SX,PX,DX,GROUPS}
    ox, co, n = I
    @inbounds begin
        KX, CIN = size(w, 1), size(w, 2)
        A = accum(eltype(x))
        acc = bias === nothing ? zero(A) : A(bias[co])
        g = GROUPS == 1 ? 0 : (co - 1) ÷ (size(w, 3) ÷ GROUPS)
        bx = (ox - 1) * SX - PX
        for ci in 1:CIN
            xc = g * CIN + ci
            for kx in 1:KX
                ix = bx + (kx - 1) * DX + 1
                (ix < 1 || ix > size(x, 1)) && continue
                acc = muladd(A(x[ix, xc, n]), A(w[kx, ci, co]), acc)
            end
        end
        acc
    end
end

convolution1d!(out, x, w, bias, stride, padding, dilation, groups) =
    launch!(conv1d, out, x, w, bias, Val(stride[1]), Val(padding[1]),
            Val(dilation[1]), Val(groups))
