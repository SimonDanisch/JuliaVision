# What does `mattewidth` actually buy? Cost and edge sharpness per resolution.
#
# Two traps this harness exists to avoid:
#  * every width is a new set of tensor shapes, so each arm pays its own kernel
#    compilation. Timing the whole arm measured the compiler. Per-frame deltas
#    come off the progress callback and the settled median is reported.
#  * comparing arm to arm measures TRACKING divergence, not sharpness. The
#    propagator is causal, so two resolutions follow the subject differently and
#    that difference is an order of magnitude larger than any edge effect
#    (measured: mean |dalpha| of 0.23-0.33 inside the band, and non-monotonic in
#    width). Sharpness is therefore measured on each matte ALONE, as the
#    thickness of its own transition band once upsampled to the layer.
using VideoEditor, MatAnyoneRunner, Lava, Printf, Statistics
import VideoEditor as VE

"Bilinear upsample of one alpha plane to (W,H) — what `applymatte!` samples."
function upsample(a::AbstractMatrix{UInt8}, W::Int, H::Int)
    w, h = size(a); out = Matrix{Float32}(undef, W, H)
    @inbounds for j in 1:H, i in 1:W
        u = (i - 0.5f0) * w / W + 0.5f0; v = (j - 0.5f0) * h / H + 0.5f0
        x0 = clamp(floor(Int, u), 1, w); x1 = min(x0 + 1, w)
        y0 = clamp(floor(Int, v), 1, h); y1 = min(y0 + 1, h)
        fx = clamp(u - x0, 0f0, 1f0); fy = clamp(v - y0, 0f0, 1f0)
        out[i,j] = ((Float32(a[x0,y0])*(1-fx) + Float32(a[x1,y0])*fx)*(1-fy) +
                    (Float32(a[x0,y1])*(1-fx) + Float32(a[x1,y1])*fx)*fy) / 255
    end
    out
end

"""
Mean transition width in LAYER pixels: band area over boundary length. A matte
computed at the layer's own resolution resolves the thinnest band it can; one
computed smaller and stretched cannot get below its own sample spacing.
"""
function bandthickness(alpha, W::Int, H::Int)
    tot = 0.0; nf = 0
    for k in axes(alpha, 3)
        up = upsample(@view(alpha[:,:,k]), W, H)
        band = count(x -> 0.05f0 < x < 0.95f0, up)
        # boundary length ~ number of 4-neighbour crossings of the 0.5 level
        edge = 0
        @inbounds for j in 1:H-1, i in 1:W-1
            c = up[i,j] > 0.5f0
            (c != (up[i+1,j] > 0.5f0)) && (edge += 1)
            (c != (up[i,j+1] > 0.5f0)) && (edge += 1)
        end
        edge == 0 && continue
        tot += 2band / edge; nf += 1
    end
    nf == 0 ? NaN : tot / nf
end

videopath  = "/windows/Users/sdani/Cloudi/giffers/20260708_160827.mp4"
seedpath   = "/tmp/birdseed.bin"
nframes    = 60
widths     = [320, 480, 640, 756]
settleskip = 10          # frames dropped before taking the median

src  = VE.VideoSource(videopath)
seed = open(seedpath) do io
    w = read(io, Int32); h = read(io, Int32); read!(io, Matrix{UInt8}(undef, w, h))
end
VE.registermatte!(MatAnyoneRunner.matanyonepropagator())
engine = VE.FxEngine(Lava.LavaBackend())
mkclip() = (c = VE.Clip(src; src_in = 1500, src_out = 1500 + nframes - 1);
            c.crop = (0.15000000596046448, 0.25, 0.7000000178813934, 0.5); c)
LW, LH = VE.mattelayersize(mkclip())
@printf("layer %dx%d — the matte is upsampled to this when applied\n", LW, LH)
@printf("%d frames per arm, median over frames %d+\n\n", nframes, settleskip + 1)

"Run one arm; return (alpha, settled ms/frame)."
function arm(w)
    c = mkclip()
    reader = VE.framereader(c, engine; width = w)
    times = Float64[]; last = Ref(time_ns())
    track = VE.analyzematte!(c, reader, Dict(1500 => seed); mattewidth = w,
                             progress = (d, t) -> (push!(times, (time_ns() - last[]) / 1e6);
                                                   last[] = time_ns()))
    track.alpha, median(times[min(settleskip + 1, end):end])
end

arm(240)   # warm the shared code paths once, off the record
res = Dict{Int,Any}()
for w in widths
    a, ms = arm(w)
    res[w] = a
    @printf("width %4d  matte %-12s %7.1f ms/frame   alpha %6.1f MB\n",
            w, string(size(a)[1:2]), ms, prod(size(a)) / 2^20)
end

@printf("\nedge sharpness — mean transition width in layer pixels (lower = sharper):\n")
for w in widths
    @printf("  width %4d:  %.2f px\n", w, bandthickness(res[w], LW, LH))
end

# Negative control: the propagator has no RNG, so a repeat at one width must
# reproduce exactly. If it does not, every number above is noise and the
# non-monotonic cross-arm comparison had a second explanation.
a2, _ = arm(480)
@printf("\ncontrol — 480 rerun reproduces exactly: %s\n", a2 == res[480])
