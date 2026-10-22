# Port of OIDN v2.5.1's HIP inference path: devices/hip/ck_conv_wmma.cpp (CK WMMA
# grouped convolution), core/unet_filter.cpp graph, pool/upsample/concat handling.
# Copyright 2018-2025 Intel Corporation; SPDX-License-Identifier: Apache-2.0.
# Modified: KernelInterface kernels shared by Vulkan, ROCm and Metal.
#
# Layouts follow upstream HIP: activations HWC with channels padded to 8, weights
# OHWI. As a Julia array an activation is `[c, x, y]`. The convolution is the GEMM
# `Y[o, p] = Σ_k W[o, k] X[k, p]` with `k = tap * C + c`, so both operands are
# contiguous along k: weights are row-major `(o, k)`, activations column-major
# `(k, p)`. That is the layout RDNA3's WMMA loads a fragment from in one go.
#
# Activations and weights are READ through a 16-byte vector view of the same
# buffer (`stagevec`): eight halves where the backend has 8-wide vectors, a
# `uvec4` where SPIR-V stops at four components, which the cooperative-matrix
# load reads back as fp16 out of shared memory. The convolution's matrix store
# writes the scalar `Float16` view.
using Mantle.KI: AcceleratedMatrix, MatrixA, MatrixB, Accumulator

const HalfVec{V} = NTuple{V, VecElement{Float16}}
const UVec4 = NTuple{4, VecElement{UInt32}}
# Sixteen bytes either way. As 4 x fp16 on Lava, RADV read the shared tiles in
# 8-byte `ds_load_b64`s and the gather in 8-byte global loads; as a uvec4 both
# are 16-byte, as on ROCm.
stagevec(::Mantle.LavaBackend) = UVec4
stagevec(backend) = HalfVec{8}
# How the K-loop prefetches; see the convolution kernel.
prefetch(::Mantle.LavaBackend) = Val(:always)
prefetch(backend) = Val(:branch)
# The max pool needs fp16 lanes, and SPIR-V vectors stop at four.
poolvec(::Mantle.LavaBackend) = HalfVec{4}
poolvec(backend) = HalfVec{8}
halves(::Type{HalfVec{V}}) where {V} = V
halves(::Type{UVec4}) = 8
const CONV_BK = 32      # reduction step (CK KPerBlock)
const CONV_LD = 40      # shared leading dimension in halves: 32 + 8 padding
const CONV_PIXELS = 256 # pixel capacity granularity: the widest workgroup tile
const CONV_OPAD = 32    # output channels are padded to this; a 3-channel head to 16

@inline zerovec(::Type{NTuple{N, VecElement{T}}}) where {N, T} = ntuple(_ -> VecElement(zero(T)), Val(N))

# Vector index of the eight channels `c..c+7` of tap `(kx, ky)` for output pixel
# `(x, y)` in the virtual concatenation `[upsample?(src0); src1]`.
@inline function gatheraddr(x, y, k, ::Val{V}, ::Val{W}, ::Val{H}, ::Val{C0}, ::Val{C1}, ::Val{UP}) where {V, W, H, C0, C1, UP}
    CT = Int32(C0 + C1)
    tap = k ÷ CT; c = k - tap * CT
    ky = tap ÷ Int32(3); kx = tap - ky * Int32(3)
    sx = x + kx - Int32(1); sy = y + ky - Int32(1)
    # Branch-free on purpose: `&&`, a chained comparison or `?:` here becomes
    # control flow in Lava's SPIR-V, RADV then executes each guarded load under
    # an exec mask and must `s_waitcnt vmcnt(0)` before the other side zeroes
    # the same registers, so the eight prefetch loads of a step serialise.
    ok = (tap < Int32(9)) & (sx >= Int32(0)) & (sx < Int32(W)) & (sy >= Int32(0)) & (sy < Int32(H))
    # Clamped rather than zeroed when out of range (the load stays in bounds
    # either way), and the address in `UInt32`, as GLSL writes it: Julia's
    # `Int32` arithmetic wraps, so LLVM cannot prove a product non-negative and
    # keeps `÷` an arithmetic shift and every widening a sign extension, once
    # per load. Unsigned, they are a logical shift and a zero extension.
    ux = UInt32(clamp(sx, Int32(0), Int32(W - 1))); uy = UInt32(clamp(sy, Int32(0), Int32(H - 1)))
    uc = UInt32(c)
    first = (C1 == 0) | (c < Int32(C0))
    i0 = UP ? uc + UInt32(C0) * ((ux >> UInt32(1)) + UInt32(W ÷ 2) * (uy >> UInt32(1))) :
              uc + UInt32(C0) * (ux + UInt32(W) * uy)
    i1 = (uc - UInt32(C0)) + UInt32(max(C1, 1)) * (ux + UInt32(W) * uy)
    return UInt32(1) + ifelse(first, i0, i1) ÷ UInt32(V), ok, first
end

# One load from the clamped address; the zero padding is applied where the
# value is staged into shared memory (`padded`), a K-step later. Selecting right
# after the load made ACO wait for it there (`s_waitcnt vmcnt`), before the
# step's products, so the prefetch never overlapped the matrix work.
# Choosing the SOURCE stays a branch with a load on either side. An `ifelse`
# between the two arrays is a select of structs whose pointer field Lava cannot
# extract, and a load on both sides costs no wait.
@inline function gather(X0, X1, i, first, ::Val{C1}, ::Val{j}) where {C1, j}
    return if C1 == 0 || first   # folds away for a single source
        @inbounds X0[i + UInt32(j)]
    else
        @inbounds X1[i + UInt32(j)]
    end
end

@inline padded(v::T, ok::Bool) where {T} = ifelse(ok, v, zerovec(T))

@inline function relu16(m::AcceleratedMatrix{Float32, MT, MT, Accumulator}) where {MT}
    for i in Int32(0):(KI.coopmat_length(AcceleratedMatrix{Float32, MT, MT, Accumulator}) - Int32(1))
        m = KI.coopmat_setcomp(m, i, max(KI.coopmat_getcomp(m, i), 0.0f0))
    end
    return m
end

"""
    ConvTiling(BO, BP, WO, WP)

A workgroup computes `BO` output channels for `BP` pixels with `WO x WP` wave32
subgroups, each owning a `(BO/WO) x (BP/WP)` block of device-sized accumulators
(16x16 on Vulkan/ROCm, 8x8 on Metal).

Upstream's CK instance is `ConvTiling(32, 256, 2, 2)`: 16 channels x 128 pixels
per wave. On gfx1151 a 32 x 64 wave block is 20-45% faster, because it loads six
fragments per eight products instead of nine, and a workgroup that owns all of a
layer's output channels gathers each input pixel once instead of `CO/32` times.
"""
const ConvTiling = NTuple{4, Int}
const CONV_TILINGS = ConvTiling[(32, 256, 2, 2), (16, 128, 1, 2), (32, 128, 1, 2), (64, 256, 2, 4),
                                (96, 256, 3, 4), (128, 256, 4, 4), (192, 128, 6, 2), (128, 64, 4, 1),
                                (16, 256, 1, 4), (64, 128, 2, 2), (96, 128, 3, 2), (96, 64, 3, 1),
                                (128, 128, 4, 2), (192, 64, 6, 1)]
const CONV_KERNELS = Dict{ConvTiling, Any}()
const CONV_KERNELS_8 = Dict{ConvTiling, Any}()
convthreads(t::ConvTiling) = 32 * t[3] * t[4]

@inline convaccinit(bias, ptr, off, ::Val{MT}) where {MT} =
    convert(AcceleratedMatrix{Float32,MT,MT,Accumulator},
        AcceleratedMatrix{eltype(bias),MT,MT,Accumulator}(ptr,off,0))

function convmatrixtile(backend)
    c = DNNKernels.caps(backend)
    c.coopmat && c.tile in (8,16) || error("OIDN needs 8x8 or 16x16 cooperative matrices")
    return c.tile
end

for MT in (8,16), cfg in CONV_TILINGS
    BO, BP, WO, WP = cfg
    NT = 32 * WO * WP
    TO = BO ÷ WO ÷ MT; TP = BP ÷ WP ÷ MT
    KG = CONV_BK ÷ 8                               # 8-half groups per k-row
    AV = cld(BO * KG, NT); BV = cld(BP * KG, NT)   # groups staged per thread
    RS = NT ÷ KG                                   # rows between a thread's groups
    AG = AV * NT != BO * KG; BG = BV * NT != BP * KG   # the last group needs a guard
    kname = MT == 16 ? Symbol("conv3x3_", BO, "x", BP, "_", WO, "x", WP, "!") :
        Symbol("conv3x3_", BO, "x", BP, "_", WO, "x", WP, "_tile8!")
    @eval begin
        """
        One 3x3 convolution (padding 1) plus bias and optional ReLU, fp16 in and
        out, fp32 accumulation, tiled `$($cfg)`. `X1` is the concatenated second
        source (`C1 > 0`); `UP` reads `X0` at half resolution, which is the
        decoder's nearest upsample. `Wt` is `(K, CO)` with `K = 9 * (C0 + C1)`
        rounded up to `CONV_BK`. `X0`, `X1` and `Wt` are `stagevec` views of `V` halves; `Y`
        is scalar `Float16`.
        """
        function $kname(Y, X0, X1, Wt, bias, ::Val{V}, ::Val{W}, ::Val{H}, ::Val{C0}, ::Val{C1},
                        ::Val{UP}, ::Val{CO}, ::Val{K}, ::Val{RELU}, ::Val{PF} = Val(:branch)) where {V, W, H, C0, C1, UP, CO, K, RELU, PF}
            T = eltype(X0)
            NV = 8 ÷ V                 # vectors per eight staged halves
            LDV = CONV_LD ÷ V
            sA = KI.localmemory(T, Val((LDV * $BO,)), Val(1))
            sB = KI.localmemory(T, Val((LDV * $BP,)), Val(2))
            # Vulkan consumes vector-packed storage directly; Metal's matrix
            # load requires a scalar half view and scalar offsets/strides.
            matA = $MT == 8 ? reinterpret(Float16, sA) : sA
            matB = $MT == 8 ? reinterpret(Float16, sB) : sB
            matV = $MT == 8 ? 1 : V
            matLD = $MT == 8 ? CONV_LD : LDV
            tid = Int32(KI.get_local_id().x - 1)
            blk = Int32(KI.get_group_id().x - 1)
            nO = Int32(CO ÷ $BO)
            o0 = (blk % nO) * Int32($BO)
            p0 = (blk ÷ nO) * Int32($BP)
            s = tid ÷ Int32(32)
            wo = (s % Int32($WO)) * Int32($(BO ÷ WO))
            wp = (s ÷ Int32($WO)) * Int32($(BP ÷ WP))
            kq = tid % Int32($KG)      # which 8 of the step's k this thread stages
            pp = tid ÷ Int32($KG)      # its first row, then every RS-th
            M = Int32(W * H)
            Base.Cartesian.@nexprs $BV r -> begin
                P_r = p0 + min(pp + Int32($RS * (r - 1)), Int32($BP - 1))
                y_r = P_r ÷ Int32(W); x_r = P_r - y_r * Int32(W)
                y_r = ifelse(P_r < M, y_r, Int32(-2))   # a padding pixel reads only zeros
            end
            wa = Int32(1) + (kq * Int32(8) + (o0 + pp) * Int32(K)) ÷ Int32(V)
            Base.Cartesian.@nexprs $AV r -> (wr_r = min(pp + Int32($RS * (r - 1)), Int32($BO - 1)) - pp)

            bp = pointer(bias)
            Base.Cartesian.@nexprs $TO i -> Base.Cartesian.@nexprs $TP j ->
                c_i_j = convaccinit(bias, bp, Int32(1) + o0 + wo + Int32($MT * (i - 1)), Val($MT))

            # Prologue: step zero into registers.
            Base.Cartesian.@nexprs $BV r -> begin
                (g_r, ok_r, f_r) = gatheraddr(x_r, y_r, kq * Int32(8), Val(V), Val(W), Val(H), Val(C0), Val(C1), Val(UP))
                Base.Cartesian.@nexprs 2 q -> (q <= NV && (b_r_q = gather(X0, X1, g_r, f_r, Val(C1), Val(q - 1))))
            end
            Base.Cartesian.@nexprs $AV r -> Base.Cartesian.@nexprs 2 q ->
                (q <= NV && (a_r_q = @inbounds Wt[wa + wr_r * Int32(K ÷ V) + Int32(q - 1)]))

            nk = Int32(K ÷ CONV_BK)
            for kb in Int32(0):(nk - Int32(1))
                Base.Cartesian.@nexprs $AV r -> Base.Cartesian.@nexprs 2 q ->
                    (q <= NV && (!$AG || r < $AV || pp + Int32($RS * (r - 1)) < Int32($BO)) &&
                     (@inbounds sA[Int32(1) + kq * Int32(NV) + (pp + Int32($RS * (r - 1))) * Int32(LDV) + Int32(q - 1)] = a_r_q))
                Base.Cartesian.@nexprs $BV r -> Base.Cartesian.@nexprs 2 q ->
                    (q <= NV && (!$BG || r < $BV || pp + Int32($RS * (r - 1)) < Int32($BP)) &&
                     (@inbounds sB[Int32(1) + kq * Int32(NV) + (pp + Int32($RS * (r - 1))) * Int32(LDV) + Int32(q - 1)] = padded(b_r_q, ok_r)))
                KI.barrier()
                # Issue the next step's global reads before this step's products.
                # `PF = :always` (Lava): unconditionally, the last step reloading
                # itself, which nothing stores. Under `if kb + 1 < nk` the
                # prefetched registers are phis at the merge, ACO resolves them
                # with copies at the end of the branch, and a copy of a loaded
                # value waits for the load, so the prefetch never overlapped the
                # products on RADV. LLVM coalesces the phis, so ROCm keeps the
                # branch and skips the redundant loads (`PF = :branch`).
                if PF === :always || kb + Int32(1) < nk
                    kn = min(kb + Int32(1), nk - Int32(1))
                    k = kn * Int32(CONV_BK) + kq * Int32(8)
                    Base.Cartesian.@nexprs $BV r -> begin
                        (g_r, ok_r, f_r) = gatheraddr(x_r, y_r, k, Val(V), Val(W), Val(H), Val(C0), Val(C1), Val(UP))
                        Base.Cartesian.@nexprs 2 q -> (q <= NV && (b_r_q = gather(X0, X1, g_r, f_r, Val(C1), Val(q - 1))))
                    end
                    wk = wa + kn * Int32(CONV_BK ÷ V)
                    Base.Cartesian.@nexprs $AV r -> Base.Cartesian.@nexprs 2 q ->
                        (q <= NV && (a_r_q = @inbounds Wt[wk + wr_r * Int32(K ÷ V) + Int32(q - 1)]))
                end
                Base.Cartesian.@nexprs $(CONV_BK ÷ MT) u -> begin
                    Base.Cartesian.@nexprs $TO i -> (a_i = AcceleratedMatrix{Float16, $MT, $MT, MatrixA}(
                        matA, Int32(1) + Int32($MT * (u - 1) ÷ matV) + (wo + Int32($MT * (i - 1))) * Int32(matLD), Int32(matLD), Val(true)))
                    Base.Cartesian.@nexprs $TP j -> begin
                        b = AcceleratedMatrix{Float16, $MT, $MT, MatrixB}(
                            matB, Int32(1) + Int32($MT * (u - 1) ÷ matV) + (wp + Int32($MT * (j - 1))) * Int32(matLD), Int32(matLD))
                        Base.Cartesian.@nexprs $TO i -> (c_i_j = muladd(a_i, b, c_i_j))
                    end
                end
                KI.barrier()
            end

            Base.Cartesian.@nexprs $TO i -> Base.Cartesian.@nexprs $TP j -> begin
                v = RELU ? relu16(c_i_j) : c_i_j
                h = convert(AcceleratedMatrix{Float16, $MT, $MT, Accumulator}, v)
                copyto!(pointer(Y), Int32(1) + o0 + wo + Int32($MT * (i - 1)) + (p0 + wp + Int32($MT * (j - 1))) * Int32(CO),
                        Int32(CO), h)
            end
            return nothing
        end
        ($MT == 16 ? CONV_KERNELS : CONV_KERNELS_8)[$cfg] = $kname
    end
end

# Fastest per output-channel count on gfx1151, every layer of the large U-Net at
# 1080p, best of three (`BlackholeImage/tools/oidn_conv_sweep.jl`). A workgroup
# owning all output channels wins wherever it fits; 31-35 TFLOPS from 64 up.
const CONV_TABLE = Dict{Int, ConvTiling}(16 => (16, 128, 1, 2), 32 => (32, 128, 1, 2), 64 => (64, 256, 2, 4),
                                         96 => (96, 256, 3, 4), 128 => (128, 256, 4, 4), 192 => (192, 128, 6, 2),
                                         256 => (128, 64, 4, 1))
convtiling(CO) = get(CONV_TABLE, CO, (32, 256, 2, 2))

"""2x2 max pool, HWC, one `poolvec` of `V` halves per invocation; `Y` and `X` are vector views."""
function maxpool_kernel!(Y, X, ::Val{V}, ::Val{W}, ::Val{H}, ::Val{C}) where {V, W, H, C}
    g = Int32(KI.get_global_id().x - 1)
    CV = Int32(C ÷ V); Wo = Int32(W ÷ 2); Ho = Int32(H ÷ 2)
    g < CV * Wo * Ho || return nothing
    q = g % CV; p = g ÷ CV
    y = p ÷ Wo; x = p - y * Wo
    i00 = Int32(1) + q + CV * (Int32(2) * x + Int32(W) * Int32(2) * y)
    i10 = i00 + CV; i01 = i00 + CV * Int32(W); i11 = i01 + CV
    @inbounds a = X[i00]; @inbounds b = X[i10]; @inbounds c = X[i01]; @inbounds d = X[i11]
    @inbounds Y[Int32(1) + q + CV * p] = ntuple(e -> VecElement(max(max(a[e].value, b[e].value), max(c[e].value, d[e].value))), Val(V))
    return nothing
end

# ── input / output processing (core/input_process.h, output_process.h) ────────

# A source image on the device: Float32, element `(c, x, y)` at
# `1 + c*sc + x*sx + y*sy`. Interleaved RGB and planar `(w, h, 3)` are both this.
struct ImageStrides
    sc::Int32
    sx::Int32
    sy::Int32
end
interleaved(w) = ImageStrides(1, 3, 3w)
planarstrides(w, h) = ImageStrides(w * h, 1, w)
@inline pixel(a, s::ImageStrides, c, x, y) = @inbounds a[Int32(1) + c * s.sc + x * s.sx + y * s.sy]
@inline ksanitize(v, lo, hi) = clamp(isnan(v) ? 0.0f0 : v, lo, hi)

"""
Write one tile's network input, HWC with `CI` channels (zero padded), from the
full-resolution images. The image region sits at `(ax, ay)` inside the tile and
everything else is zero, as in the host path this replaces.
"""
function input_kernel!(I, src, g1, g2, ::Val{MODE}, ::Val{CI}, ::Val{TW}, ::Val{TH},
                       ax, ay, x0, y0, rw, rh, ss::ImageStrides, s1::ImageStrides, s2::ImageStrides,
                       scale) where {MODE, CI, TW, TH}
    g = Int32(KI.get_global_id().x - 1)
    g < Int32(TW * TH) || return nothing
    y = g ÷ Int32(TW); x = g - y * Int32(TW)
    ix = x - ax; iy = y - ay
    inside = Int32(0) <= ix < rw && Int32(0) <= iy < rh
    px = ifelse(inside, x0 + ix, Int32(0)); py = ifelse(inside, y0 + iy, Int32(0))
    o = Int32(1) + Int32(CI) * g
    for c in Int32(0):Int32(CI - 1)
        v = 0.0f0
        if inside && c < Int32(MODE === :hdr ? 9 : 3)
            v = if MODE === :albedo
                srgb_forward(ksanitize(pixel(src, ss, c, px, py), 0.0f0, 1.0f0))
            elseif MODE === :normal
                ksanitize(pixel(src, ss, c, px, py), -1.0f0, 1.0f0) * 0.5f0 + 0.5f0
            elseif c < Int32(3)
                pu_forward(ksanitize(pixel(src, ss, c, px, py) * scale, 0.0f0, floatmax(Float32))) / PU_MAX
            elseif c < Int32(6)
                ksanitize(pixel(g1, s1, c - Int32(3), px, py), 0.0f0, 1.0f0)
            else
                ksanitize(pixel(g2, s2, c - Int32(6), px, py), -1.0f0, 1.0f0) * 0.5f0 + 0.5f0
            end
        end
        @inbounds I[o + c] = Float16(v)
    end
    return nothing
end

"""Write a tile's interior from the network output `Y` (HWC, `CO` channels)."""
function output_kernel!(dst, Y, ::Val{MODE}, ::Val{CO}, ::Val{TW}, ax, ay, x0, y0,
                        left, top, nx, ny, sd::ImageStrides, invscale) where {MODE, CO, TW}
    g = Int32(KI.get_global_id().x - 1)
    g < nx * ny || return nothing
    j = g ÷ nx; i = g - j * nx
    ix = left + i; iy = top + j
    t = Int32(1) + Int32(CO) * ((ax + ix) + Int32(TW) * (ay + iy))
    d = Int32(1) + (x0 + ix) * sd.sx + (y0 + iy) * sd.sy
    for c in Int32(0):Int32(2)
        v = max(Float32(@inbounds Y[t + c]), 0.0f0)
        r = if MODE === :albedo
            clamp(srgb_inverse(v), 0.0f0, 1.0f0)
        elseif MODE === :normal
            clamp(v * 2.0f0 - 1.0f0, -1.0f0, 1.0f0)
        else
            pu_inverse(v * PU_MAX) * invscale
        end
        @inbounds dst[d + c * sd.sc] = r
    end
    return nothing
end

# ── the network (core/unet_filter.cpp, large variant) ──────────────────────────

struct ConvLayer
    C0::Int        # channels read from the first source (padded)
    C1::Int        # concatenated second source, 0 if none
    up::Bool       # first source is upsampled 2x
    CO::Int        # output channels, padded to CONV_OPAD
    K::Int
    Wt::Any        # (K, CO) Float16 on the device
    bias::Any      # CO Float32 on the device
end

const LEVEL_CHANNELS = (64, 96, 128, 192, 256)   # ec1..ec5 = dc1..dc4 of "large"
padc(c) = cld(c, 8) * 8

"Pack `(kx, ky, cin, cout)` weights for the convolution kernels: `k = tap*(C0+C1) + c`."
function packconv(w::AbstractArray{<:Real, 4}, b, C0real, C0, C1real, C1, CO)
    CT = C0 + C1
    K = cld(9 * CT, CONV_BK) * CONV_BK
    Wt = zeros(Float16, K, CO)
    bias = zeros(Float32, CO)
    for o in 1:size(w, 4), ky in 0:2, kx in 0:2
        tap = ky * 3 + kx
        for c in 1:C0real
            Wt[1 + tap * CT + (c - 1), o] = w[kx + 1, ky + 1, c, o]
        end
        for c in 1:C1real
            Wt[1 + tap * CT + C0 + (c - 1), o] = w[kx + 1, ky + 1, C0real + c, o]
        end
        bias[o] = b[o]
    end
    return Wt, bias, K
end

function devicecopy(backend, a)
    d = KA.allocate(backend, eltype(a), size(a))
    copyto!(d, a)
    return d
end

"""
    UNet(weights, cin, cout, backend)

The large OIDN U-Net as nineteen `ConvLayer`s. `cin`/`cout` are the real input
and output channel counts; inputs are padded to 8 channels, the output to 32.
"""
struct UNet
    cin::Int
    CI::Int
    layers::Dict{Symbol, ConvLayer}
end
function UNet(weights::AbstractDict, cin::Int, backend)
    CI = padc(cin)
    layers = Dict{Symbol, ConvLayer}()
    function add!(name, C0real, C0, C1real, C1, up)
        w = weights[string(name, ".weight")]; b = weights[string(name, ".bias")]
        CO = size(w, 4) <= 16 ? 16 : cld(size(w, 4), CONV_OPAD) * CONV_OPAD
        Wt, bias, K = packconv(w, b, C0real, C0, C1real, C1, CO)
        layers[name] = ConvLayer(C0, C1, up, CO, K, devicecopy(backend, vec(Wt)), devicecopy(backend, bias))
    end
    e = LEVEL_CHANNELS
    add!(:enc_conv1a, cin, CI, 0, 0, false); add!(:enc_conv1b, e[1], e[1], 0, 0, false)
    add!(:enc_conv2a, e[1], e[1], 0, 0, false); add!(:enc_conv2b, e[2], e[2], 0, 0, false)
    add!(:enc_conv3a, e[2], e[2], 0, 0, false); add!(:enc_conv3b, e[3], e[3], 0, 0, false)
    add!(:enc_conv4a, e[3], e[3], 0, 0, false); add!(:enc_conv4b, e[4], e[4], 0, 0, false)
    add!(:enc_conv5a, e[4], e[4], 0, 0, false); add!(:enc_conv5b, e[5], e[5], 0, 0, false)
    add!(:dec_conv4a, e[5], e[5], e[3], e[3], true); add!(:dec_conv4b, e[4], e[4], 0, 0, false)
    add!(:dec_conv3a, e[4], e[4], e[2], e[2], true); add!(:dec_conv3b, e[3], e[3], 0, 0, false)
    add!(:dec_conv2a, e[3], e[3], e[1], e[1], true); add!(:dec_conv2b, e[2], e[2], 0, 0, false)
    add!(:dec_conv1a, e[2], e[2], cin, CI, true); add!(:dec_conv1b, e[1], e[1], 0, 0, false)
    add!(:dec_conv1c, e[1], e[1], 0, 0, false)
    return UNet(cin, CI, layers)
end

"""
Activations for one tile size: per level a pair for the two convolutions and
the pooled skip connection the decoder concatenates. Pixel capacity is rounded
up to `CONV_PIXELS`, the widest workgroup tile, which the last tile writes into.
"""
struct TileBuffers
    tw::Int
    th::Int
    S::DataType           # convolution staging vector, `stagevec`
    P::DataType           # pool vector, `poolvec`
    input::Any            # 16 channels (HDR) / 8 (auxiliary), level 0
    output::Any           # 32 channels, level 0
    a::Vector{Any}        # per level
    b::Vector{Any}
    pool::Vector{Any}     # pool[l] is level l's input, from level l-1
end
capacity(tw, th, l) = cld((tw >> l) * (th >> l), CONV_PIXELS) * CONV_PIXELS
function TileBuffers(backend, tw, th)
    alloc(c, l) = KA.allocate(backend, Float16, c * capacity(tw, th, l))
    e = LEVEL_CHANNELS
    a = Any[alloc(e[l + 1], l) for l in 0:4]
    b = Any[alloc(e[l + 1], l) for l in 0:4]
    pool = Any[nothing; [alloc(e[l], l) for l in 1:4]]
    return TileBuffers(tw, th, stagevec(backend), poolvec(backend), alloc(16, 0), alloc(CONV_OPAD, 0), a, b, pool)
end

function conv!(backend, t::TileBuffers, L::ConvLayer, Y, X0, X1, level; tiling = convtiling(L.CO), pf = prefetch(backend))
    W = t.tw >> level; H = t.th >> level
    BO, BP = tiling
    nblk = (L.CO ÷ BO) * (capacity(t.tw, t.th, level) ÷ BP)
    S = t.S; nt = convthreads(tiling)
    kernels = convmatrixtile(backend) == 8 ? CONV_KERNELS_8 : CONV_KERNELS
    Mantle.kilaunch!(backend, kernels[tiling], Y, reinterpret(S, X0), reinterpret(S, X1 === nothing ? X0 : X1),
        reinterpret(S, L.Wt), L.bias, Val(halves(S)), Val(W), Val(H), Val(L.C0), Val(L.C1), Val(L.up), Val(L.CO),
        Val(L.K), Val(true), pf; ndrange = nblk * nt, workgroupsize = nt)
    return Y
end
function pool!(backend, t::TileBuffers, Y, X, C, level)
    W = t.tw >> level; H = t.th >> level
    V = halves(t.P)
    n = (C ÷ V) * (W ÷ 2) * (H ÷ 2)
    Mantle.kilaunch!(backend, maxpool_kernel!, reinterpret(t.P, Y), reinterpret(t.P, X), Val(V), Val(W), Val(H), Val(C);
        ndrange = cld(n, 256) * 256, workgroupsize = 256)
    return Y
end

"Run the U-Net on `t.input`; the result is `t.output`."
function forward!(backend, net::UNet, t::TileBuffers)
    L = net.layers; a = t.a; b = t.b; p = t.pool; e = LEVEL_CHANNELS
    conv!(backend, t, L[:enc_conv1a], a[1], t.input, nothing, 0)
    conv!(backend, t, L[:enc_conv1b], b[1], a[1], nothing, 0); pool!(backend, t, p[2], b[1], e[1], 0)
    conv!(backend, t, L[:enc_conv2a], a[2], p[2], nothing, 1)
    conv!(backend, t, L[:enc_conv2b], b[2], a[2], nothing, 1); pool!(backend, t, p[3], b[2], e[2], 1)
    conv!(backend, t, L[:enc_conv3a], a[3], p[3], nothing, 2)
    conv!(backend, t, L[:enc_conv3b], b[3], a[3], nothing, 2); pool!(backend, t, p[4], b[3], e[3], 2)
    conv!(backend, t, L[:enc_conv4a], a[4], p[4], nothing, 3)
    conv!(backend, t, L[:enc_conv4b], b[4], a[4], nothing, 3); pool!(backend, t, p[5], b[4], e[4], 3)
    conv!(backend, t, L[:enc_conv5a], a[5], p[5], nothing, 4)
    conv!(backend, t, L[:enc_conv5b], b[5], a[5], nothing, 4)
    conv!(backend, t, L[:dec_conv4a], a[4], b[5], p[4], 3)
    conv!(backend, t, L[:dec_conv4b], b[4], a[4], nothing, 3)
    conv!(backend, t, L[:dec_conv3a], a[3], b[4], p[3], 2)
    conv!(backend, t, L[:dec_conv3b], b[3], a[3], nothing, 2)
    conv!(backend, t, L[:dec_conv2a], a[2], b[3], p[2], 1)
    conv!(backend, t, L[:dec_conv2b], b[2], a[2], nothing, 1)
    conv!(backend, t, L[:dec_conv1a], a[1], b[2], t.input, 0)
    conv!(backend, t, L[:dec_conv1b], b[1], a[1], nothing, 0)
    conv!(backend, t, L[:dec_conv1c], t.output, b[1], nothing, 0)
    return t.output
end
