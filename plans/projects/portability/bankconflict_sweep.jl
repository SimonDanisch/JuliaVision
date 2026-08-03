# epad x rpad sweep for the cooperative-matrix flash kernel, on RDNA 3.5.
#
# The defaults in `flashepad`/`flashrpad` are measurements taken on a 32-bank,
# 4-byte, wave32 NVIDIA part. This machine's LDS is also 32 banks of 4 bytes
# (measured, REPORT.md "RDNA 3.5's LDS has 32 banks"), but its wave is 64 and its
# coopmat modules are pinned back to 32, so the access pattern is not the same
# shape and the defaults have to be re-derived rather than assumed.
#
# ── The prediction this sweep is testing ──────────────────────────────────────
#
# `qs`/`kvs` are Float16 at a row stride of `EP + epad` elements. An LDS bank word
# is 4 bytes, so the stride in bank words is `(EP + epad)/2`, and the earlier
# stride sweep measured the cost to be a function of `gcd(stride, 32)` and nothing
# else. `EP` is always a multiple of 16, so:
#
#     epad   E=64: EP+epad   bank words   gcd(.,32)   predicted
#     0      64                32          32         worst (every row one bank)
#     2      66                33           1         FREE
#     4      68                34           2         near-free
#     8      72                36           4         the shipped default
#     16     80                40           8         worse than 8
#
# So the shipped `epad = 8` should NOT be optimal here: it breaks the 32-way
# collision down to 4-way, but `epad = 2` should break it entirely. That is a
# prediction made from the hardware measurement BEFORE this sweep ran, and it is
# the thing to confirm or kill. It is written down here so the sweep cannot be
# read as having found whatever it found.
#
# The NVIDIA baseline to compare against, from `flashepad`'s docstring, same
# shape (Lq = Lk = 4096, 8 head-batches, tiling 64x32/8):
#
#     E    EP    unpadded   padded(8)
#     16   16    1.785      1.782
#     32   32    2.592      2.222    -14.3%
#     48   48    3.546      3.644
#     64   64    5.266      3.614    -31.4%
#     72   80    4.384      4.404
#
# ── Method ────────────────────────────────────────────────────────────────────
#
# GUARDRAILS §6: warm-up is gated on the measurement settling, not on a fixed
# count, and every configuration is checked for correctness before it is allowed
# to contribute a time. §3: a configuration that writes nothing would otherwise
# post the best time in the table, so output coverage is asserted, not assumed.

using Lava, KernelAbstractions, DNNKernels, Statistics, Printf
const KA = KernelAbstractions

const BASELINE_NV = Dict(   # E => (unpadded, epad=8), milliseconds
    16 => (1.785, 1.782), 32 => (2.592, 2.222), 48 => (3.546, 3.644),
    64 => (5.266, 3.614), 72 => (4.384, 4.404))

attnref(q, k, v, scale) = begin
    E, L, H, B = size(q)
    o = similar(q)
    for b in 1:B, h in 1:H
        s = (transpose(view(k, :, :, h, b)) * view(q, :, :, h, b)) .* scale
        s .-= maximum(s; dims = 1)
        p = exp.(s); p ./= sum(p; dims = 1)
        o[:, :, h, b] = view(v, :, :, h, b) * p
    end
    o
end

"""
Time one configuration. Returns `(ms, ok)`; `ms` is `NaN` when the configuration
refused to launch, and `ok` is false when it launched and computed the wrong
thing — either disqualifies it from the table.
"""
function timeconfig(ctx, back, q, k, v, scale, ref, o; reps = 15, kw...)
    fill!(o, 0f0)
    launched = try
        DNNKernels.sdpaflashcm!(ctx, o, q, k, v, scale; kw...)
    catch err
        err isa Union{ErrorException,ArgumentError} || rethrow()
        false
    end
    launched || return (NaN, false, "refused")
    KA.synchronize(back)

    got = Array(o)
    # §3: a kernel that wrote nothing is fast and wrong.
    maximum(abs, got) > 1e-3 || return (NaN, false, "wrote nothing")
    err = maximum(abs, got .- ref) / maximum(abs, ref)
    err < 5e-3 || return (NaN, false, @sprintf("wrong (rel %.2e)", err))

    # Warm up until three consecutive reps agree to 2%, so the clock is up before
    # anything is recorded. Capped, so a genuinely noisy config still reports.
    last3 = Float64[]
    for _ in 1:40
        t = @elapsed begin
            DNNKernels.sdpaflashcm!(ctx, o, q, k, v, scale; kw...)
            KA.synchronize(back)
        end
        push!(last3, t * 1e3); length(last3) > 3 && popfirst!(last3)
        length(last3) == 3 && (maximum(last3) - minimum(last3)) / minimum(last3) < 0.02 && break
    end

    ts = Float64[]
    for _ in 1:reps
        t = @elapsed begin
            DNNKernels.sdpaflashcm!(ctx, o, q, k, v, scale; kw...)
            KA.synchronize(back)
        end
        push!(ts, t * 1e3)
    end
    (minimum(ts), true, @sprintf("med %.3f", median(ts)))
end

function sweep(; Es = (16, 32, 48, 64, 72), epads = (0, 2, 4, 8, 16),
                 rpads = (0, 2, 4, 8), L = 4096, H = 8, B = 1,
                 BR = 64, BC = 32, NW = 8)
    back = LavaBackend(); ctx = DNNKernels.Ctx(back); dev = ctx.dev
    @info "device" name = Lava.vk_context().device_name subgroup = dev.subgroup coopmat = dev.coopmatsubgroup tile = dev.tile
    rows = NamedTuple[]
    for E in Es
        EP = cld(E, dev.tile) * dev.tile
        qh = randn(Float32, E, L, H, B) .* 0.2f0
        kh = randn(Float32, E, L, H, B) .* 0.2f0
        vh = randn(Float32, E, L, H, B) .* 0.2f0
        f16(x) = DNNKernels.toback(back, Float16.(x))
        q, k, v = f16(qh), f16(kh), f16(vh)
        scale = Float32(1 / sqrt(E))
        ref = attnref(Float32.(Float16.(qh)), Float32.(Float16.(kh)),
                      Float32.(Float16.(vh)), scale)
        o = KA.allocate(back, Float32, E, L, H, B)
        for epad in epads, rpad in rpads
            ms, ok, note = timeconfig(ctx, back, q, k, v, scale, ref, o;
                                      BR, BC, NW, epad, rpad)
            bw = (EP + epad) ÷ 2                       # Float16 stride in bank words
            push!(rows, (; E, EP, epad, rpad, ms, ok, note,
                           bankwords = bw, gcd32 = gcd(bw, 32)))
            @printf("E=%3d EP=%3d epad=%2d rpad=%2d  bw=%3d gcd=%2d  %s  %s\n",
                    E, EP, epad, rpad, bw, gcd(bw, 32),
                    ok ? @sprintf("%8.3f ms", ms) : "       --", note)
        end
        q = k = v = o = nothing; GC.gc()
    end
    rows
end
