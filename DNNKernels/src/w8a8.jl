"""
The product both operands of which are int8, on the int8 cooperative-matrix
units.

`q8gemm` reads a packed int8 weight, dequantises it into fp16 tiles and
multiplies on the FP16 units. That is 20.8 TOP/s at Qwen-Image 2.1's shapes on
an 8060S, which is 85% of what those units can do — the fp16 ceiling measured
with staging taken out of the loop is 24.4. The int8 units measure **74.7**, and
reaching them means keeping both operands int8 all the way into the matrix.

This kernel measures **28.6 TOP/s** at `12288 x 4096 x 4224`, 1.38x the path it
replaces. Three things got it there, each measured:

  * **A is read straight from global.** A weight tile is `(M, K)` column-major,
    which is exactly a `MatrixA` load — offset `m0 + k0*M`, stride `M` — so it
    needs no staging and no barrier. Staging it in shared memory instead:
    23.1 TOP/s.
  * **B is staged as a 4-wide int8 vector, double buffered.** One 32-bit store
    per packed word instead of four byte stores (17.3 -> 19.8), a 128x128 tile
    (-> 22.4), and the next block fetched into registers while this one
    multiplies (-> 27.6 with no epilogue).
  * **The epilogue reads components by hand.** An int32 accumulator has no
    elementwise operation and no conversion to float, so each component is
    scaled through `coopmat_getcomp`/`setcomp` against a matrix of KNOWN
    coordinates — the trick `attn_flash_cm!` uses to learn the same mapping.
    It costs 27.6 -> 22.4 with the scales dropped and the same with them, so
    what it costs is the per-component intrinsics, not the scales.

The weight needs no repacking. `convrotqint8` stores four output rows to a
`UInt32` word, and four consecutive rows packed little-endian ARE int8 `(M, K)`
with M contiguous — the same bytes, reinterpreted.

## Why nothing calls it

**It is a different computation from the fp16-activation path**, not a faster
spelling of it: the activation is quantised to int8 with a per-column scale.
Against an fp32 reference of the same dequantised weight, at `1024 x 4096 x 256`:

| activation | fp16 activation | int8 activation |
| --- | --- | --- |
| Gaussian | 0.036% rms | 0.87% rms |
| 1% outliers at 8 sigma | 0.036% rms | 3.5% rms |

So the gain is 1.38x on the products, which is ~12% of a Qwen-Image 2.1 step,
and the cost is 24x to 97x the error — on a path whose current accuracy is far
better than the reference implementation's, which quantises activations too.
That is not a trade this pipeline should take, and the kernel stays off it.

What would change the answer is a quantisation with less error for the same
speed: per-group scales along k rather than one per column, which is what the
encoder's own W4A8 weights use, or keeping the outliers in fp16 alongside an
int8 bulk. Both are more kernel than this one.
"""

"A 4-wide int8 vector, i.e. `i8vec4`; see `Lava.coopmat_vecwidth`."
const W8A8V4 = NTuple{4,VecElement{Int8}}

"""One packed word as the 4-wide int8 vector a shared tile stores."""
@inline w8a8word(w::UInt32) =
    (VecElement(reinterpret(Int8, UInt8( w        & 0xff))),
     VecElement(reinterpret(Int8, UInt8((w >> 8)  & 0xff))),
     VecElement(reinterpret(Int8, UInt8((w >> 16) & 0xff))),
     VecElement(reinterpret(Int8, UInt8((w >> 24) & 0xff))))

"The output tile, the k block and the workgroup this kernel is written for."
const W8A8_BM, W8A8_BN, W8A8_BK, W8A8_WG = 128, 128, 64, 256

"""
    w8a8_quantize_kernel!(q32, scale, x, K, N, NP)

One activation column to int8 with its own scale, packed four values to a
`UInt32` along k — the layout the `MatrixB` staging reads.

A workgroup per column, because the scale is `max|x| / 127` over the whole
column and every value needs it before it can be quantised. Columns past `N`
are the GEMM's padding: they quantise to zero and contribute nothing.
"""
function w8a8_quantize_kernel!(
        q32, scale, x, ::Val{K}, N::Int32) where {K}
    red = KI.localmemory(Float32, Val((W8A8_WG,)), Val(1))
    tid = Int32(KI.get_local_id().x) - Int32(1)
    col = Int32(KI.get_group_id().x) - Int32(1)
    m = 0f0
    if col < N
        @inbounds for k in tid:Int32(W8A8_WG):Int32(K - 1)
            m = max(m, abs(Float32(x[Int32(1) + k + col * Int32(K)])))
        end
    end
    @inbounds red[tid + Int32(1)] = m
    KI.barrier()
    s = Int32(W8A8_WG ÷ 2)
    while s > Int32(0)
        @inbounds if tid < s
            red[tid + Int32(1)] = max(red[tid + Int32(1)], red[tid + s + Int32(1)])
        end
        KI.barrier()
        s ÷= Int32(2)
    end
    @inbounds a = red[1]
    inv = a == 0f0 ? 0f0 : 127f0 / a
    if tid == Int32(0)
        @inbounds scale[col + Int32(1)] = a == 0f0 ? 1f0 : a / 127f0
    end
    @inbounds for w in tid:Int32(W8A8_WG):Int32(K ÷ 4 - 1)
        word = UInt32(0)
        Base.Cartesian.@nexprs 4 r -> begin
            k = w * Int32(4) + Int32(r - 1)
            v = col < N ? round(Float32(x[Int32(1) + k + col * Int32(K)]) * inv) : 0f0
            byte = reinterpret(UInt8, unsafe_trunc(Int8, clamp(v, -127f0, 127f0)))
            word |= UInt32(byte) << (8 * (r - 1))
        end
        q32[Int32(1) + w + col * Int32(K ÷ 4)] = word
    end
    return nothing
end

"""
    w8a8_gemm_kernel!(C, A8, SA, QB, SB, Val(M), Val(N), Val(K))

`C[m, n] = (sum_k A8[m, k] * QB[k, n]) * SA[m] * SB[n]`, on the int8 units.

`A8` is the weight as int8 `(M, K)`, which is the packed weight's own bytes;
`QB` is the activation packed four values to a word along k. See the file's
docstring for where each of the three staging decisions came from.
"""
function w8a8_gemm_kernel!(
        C, A8, SA, QB, SB,
        ::Val{M}, ::Val{N}, ::Val{K}) where {M,N,K}
    sB = KI.localmemory(W8A8V4, Val((2 * W8A8_BK * W8A8_BN ÷ 4,)), Val(1))
    lsa = KI.localmemory(Float32, Val((W8A8_BM,)), Val(2))
    lsb = KI.localmemory(Float32, Val((W8A8_BN,)), Val(3))
    scoord = KI.localmemory(Float32, Val((256,)), Val(4))
    tid = Int32(KI.get_local_id().x) - Int32(1)
    blk = Int32(KI.get_group_id().x) - Int32(1)
    tm = (blk % Int32(M ÷ W8A8_BM)) * Int32(W8A8_BM)
    tn = (blk ÷ Int32(M ÷ W8A8_BM)) * Int32(W8A8_BN)
    sg = tid ÷ Int32(32)
    sm = (sg % Int32(4)) * Int32(2)
    sn = (sg ÷ Int32(4)) * Int32(4)
    pa = pointer(A8)
    @inbounds if tid < Int32(W8A8_BM)
        lsa[tid + Int32(1)] = SA[tm + tid + Int32(1)]
        lsb[tid + Int32(1)] = SB[tn + tid + Int32(1)]
    end
    # A matrix whose components are their own coordinates: `coopmat_getcomp`
    # then reports where each of this lane's components lives, which is the
    # only portable way to scale an accumulator by row and column.
    @inbounds for idx in tid:Int32(256):Int32(255)
        scoord[idx + Int32(1)] = Float32((idx % Int32(16)) + (idx ÷ Int32(16)) * Int32(16))
    end
    Base.Cartesian.@nexprs 8 i -> c_i = zero(Mantle.AcceleratedMatrix{Int32,16,16,Mantle.Accumulator})
    @inbounds Base.Cartesian.@nexprs 8 r -> begin
        idx = tid + Int32((r-1) * 256)
        sB[Int32(1) + (idx % Int32(16)) + (idx ÷ Int32(16)) * Int32(16)] =
            w8a8word(QB[Int32(1) + (idx % Int32(16)) + (tn + idx ÷ Int32(16)) * Int32(K ÷ 4)])
    end
    KI.barrier()
    nb = Int32(K ÷ W8A8_BK)
    for kb in Int32(0):(nb - Int32(1))
        cur = (kb % Int32(2)) * Int32(2048)
        k0 = kb * Int32(W8A8_BK)
        # Clamped rather than guarded: the last iteration re-fetches its own
        # block into the buffer nothing reads again, which keeps the loads and
        # both barriers at uniform control flow. Guarding them instead is a
        # WRONG ANSWER from the fourth k block on, not a slower one.
        k1 = min(kb + Int32(1), nb - Int32(1)) * Int32(W8A8_BK)
        @inbounds Base.Cartesian.@nexprs 8 r -> begin
            idx = tid + Int32((r-1) * 256)
            b_r = QB[Int32(1) + (k1 ÷ Int32(4) + idx % Int32(16)) + (tn + idx ÷ Int32(16)) * Int32(K ÷ 4)]
        end
        for kt in Int32(0):Int32(3)
            ka = k0 + kt * Int32(16)
            a1 = Mantle.AcceleratedMatrix{Int8,16,16,Mantle.MatrixA}(
                    pa, Int32(1) + (tm + sm*Int32(16)) + ka*Int32(M), Int32(M))
            a2 = Mantle.AcceleratedMatrix{Int8,16,16,Mantle.MatrixA}(
                    pa, Int32(1) + (tm + (sm+Int32(1))*Int32(16)) + ka*Int32(M), Int32(M))
            Base.Cartesian.@nexprs 4 j -> begin
                bm_j = Mantle.AcceleratedMatrix{Int8,16,16,Mantle.MatrixB}(
                        sB, Int32(1) + cur + kt*Int32(4) + (sn+Int32(j-1))*Int32(16*16), Int32(16))
                c_{2j-1} = muladd(a1, bm_j, c_{2j-1})
                c_{2j}   = muladd(a2, bm_j, c_{2j})
            end
        end
        KI.barrier()
        nxt = ((kb + Int32(1)) % Int32(2)) * Int32(2048)
        @inbounds Base.Cartesian.@nexprs 8 r -> begin
            idx = tid + Int32((r-1) * 256)
            sB[Int32(1) + nxt + (idx % Int32(16)) + (idx ÷ Int32(16)) * Int32(16)] = w8a8word(b_r)
        end
        KI.barrier()
    end
    coordmat = Mantle.AcceleratedMatrix{Float32,16,16,Mantle.Accumulator}(
                scoord, Int32(1), Int32(16), Val(false))
    Base.Cartesian.@nexprs 8 i -> begin
        co_i = unsafe_trunc(Int32, Mantle.coopmat_getcomp(coordmat, Int32(i - 1)))
        row_i = co_i % Int32(16)
        col_i = co_i ÷ Int32(16)
    end
    Base.Cartesian.@nexprs 4 j -> begin
        Base.Cartesian.@nexprs 2 t -> begin
            acc = c_{2j-2+t}
            r0 = (sm + Int32(t - 1)) * Int32(16)
            c0 = (sn + Int32(j - 1)) * Int32(16)
            f = zero(Mantle.AcceleratedMatrix{Float32,16,16,Mantle.Accumulator})
            @inbounds Base.Cartesian.@nexprs 8 i -> begin
                v = Float32(Mantle.coopmat_getcomp(acc, Int32(i - 1))) *
                    lsa[r0 + row_i + Int32(1)] * lsb[c0 + col_i + Int32(1)]
                f = Mantle.coopmat_setcomp(f, Int32(i - 1), v)
            end
            Mantle.accstore!(C, Int32(1) + (tm + r0) + (tn + c0) * Int32(M), Int32(M), f, identity)
        end
    end
    return nothing
end

"""
    w8a8ok(caps, Tout, m, k, n) -> Bool

Whether this kernel can run an `(m, k) x (k, n)` product: one tile shape, one
workgroup, and the extents it divides. Everything else keeps `q8gemm`.
"""
function w8a8ok(caps, ::Type{Tout}, m::Integer, k::Integer, n::Integer) where {Tout}
    caps.coopmat && caps.coopmatsubgroup == 32 && caps.tile == 16 || return false
    Tout in (Float16, Float32) || return false
    m % W8A8_BM == 0 && n % W8A8_BN == 0 && k % W8A8_BK == 0 || return false
    max(m * k, k * n, m * n) <= typemax(Int32)
end

"""Columns to run a W8A8 product at: `n` padded to the tile it needs."""
w8a8columns(n::Integer) = cld(n, W8A8_BN) * W8A8_BN

"""
    w8a8quantize(backend, x, np) -> (packed, scale)

Quantise `x`, an `(K, N)` activation, to int8 with a scale per column, padded
out to `np` columns of zeros. Immediate form; the declared one is in `emit.jl`.
"""
function w8a8quantize(backend, x::AbstractMatrix, np::Integer)
    K, N = size(x)
    K % 4 == 0 || throw(DimensionMismatch("W8A8 needs a multiple of four along k, got $K"))
    packed = KernelAbstractions.allocate(backend, UInt32, K ÷ 4, np)
    scale = KernelAbstractions.allocate(backend, Float32, np)
    KI.Kernel(backend, w8a8_quantize_kernel!)(packed, scale, x, Val(Int(K)), Int32(N);
                                            ndrange = np * W8A8_WG, workgroupsize = W8A8_WG)
    (packed, scale)
end

"""The weight's own bytes as int8 `(M, K)` — see this file's docstring."""
w8a8weight(A::Union{QInt8Matrix,ConvRotQInt8Matrix}) =
    # `Mantle.storage`: `A.q` is a `Mantle.Buffer` — the pool region that owns
    # the bytes — and `reinterpret` wants the array over them.
    let q = Mantle.storage(A.q)
        reshape(reinterpret(Int8, q), size(q, 1) * Q8ROWS, size(q, 2))
    end
