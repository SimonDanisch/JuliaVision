# The BasicVSRRunner README picture: five 64x64 frames of a slow pan across the
# fox, 4x upscaled, against bicubic and the frames they were cut from.
isdefined(@__MODULE__, :ExampleCommon) || include(joinpath(@__DIR__, "common.jl"))
using .ExampleCommon
using BasicVSRRunner, GPUFiltering, ColorTypes, FixedPointNumbers
import KernelAbstractions as KA

"""
MATLAB's `imresize(x, 1/s, 'bicubic')`: the cubic (a = -0.5) widened by `s` to
antialias, border replicated. REDS4's low-resolution frames are made this way,
and the model is trained to invert exactly that. An area mean is a different
degradation, and BasicVSR++ answers it with ringing and streaks.
"""
function bicubicdown(img::AbstractMatrix{RGB{Float32}}, s::Integer)
    cubic(x) = (x = abs(x); x <= 1 ? 1.5x^3 - 2.5x^2 + 1 : x < 2 ? -0.5x^3 + 2.5x^2 - 4x + 2 : 0.0)
    function pass(a)                                # along the first axis
        n = size(a, 1) ÷ s
        out = similar(a, n, size(a, 2))
        for i in 1:n
            u = (i - 0.5) * s + 0.5                 # output centre, in input pixels
            taps = floor(Int, u - 2s):ceil(Int, u + 2s)
            w = [cubic((u - t) / s) for t in taps]
            w ./= sum(w)
            for j in axes(a, 2)
                out[i, j] = sum(w[k] * a[clamp(taps[k], 1, size(a, 1)), j] for k in eachindex(taps))
            end
        end
        return out
    end
    return permutedims(pass(permutedims(pass(img))))
end

function basicvsr_examples()
    fox = RGB{Float32}.(loadimage(media("qwenimage", "fox.jpg")))    # (640, 640)
    T = 5
    # Ground truth: 256x256 windows of the face, 3 px apart; the clip is those at 1/4.
    truth = [fox[190+3k:445+3k, 90:345] for k in 0:T-1]
    small = [bicubicdown(t, 4) for t in truth]
    lqs = zeros(Float32, 64, 64, 3, T, 1)                             # (W, H, C, T, 1), 0..1
    for k in 1:T, j in 1:64, i in 1:64
        c = small[k][i, j]
        lqs[i, j, :, k, 1] .= (red(c), green(c), blue(c))
    end
    m = basicvsrppmodel()
    out = upscale(m, lqs)                                             # (256, 256, 3, T, 1) on the device
    KA.synchronize(m.backend)
    t = @elapsed (upscale(m, lqs); KA.synchronize(m.backend))
    hr = Array(out)
    k = 3                                                             # the middle frame
    sr = [RGB{Float32}(clamp.(hr[i, j, :, k, 1], 0, 1)...) for i in 1:256, j in 1:256]
    bicubic = warp!(similar(truth[k]), small[k], (0, 0, 1, 1))
    nearest = [small[k][cld(i, 4), cld(j, 4)] for i in 1:256, j in 1:256]
    saveimage(media("basicvsr", "fox.jpg"), hstrip(nearest, bicubic, sr, truth[k]))
    return (; seconds = t)
end

abspath(PROGRAM_FILE) == (@__FILE__) && basicvsr_examples()
