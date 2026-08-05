"""
`erf` and the two `gelu` forms must evaluate in `accum(T)`, not in the operand's
own precision.

## What went wrong

`erf` is Abramowitz & Stegun 7.1.26 and its docstring said "max error ~1.5e-7 —
comfortably inside fp32". That describes the *formula*. The function took
`T = typeof(float(x))`, so on a half tensor the whole approximation ran in
Float16 — and A&S 7.1.26 ends in `1 - poly*exp`, which for `|x| > 2` subtracts
two quantities that agree to four decimal digits, where fp16 has three. `gelu`
then added its own `1 + erf(...)` in half on top, cancelling again in exactly the
same range.

Both are silent: nothing overflows, nothing is NaN, and the answer is plausible
everywhere.

## What it cost

Whisper's fp16 encoder, 34 `gelu`s over 32 blocks. Against PyTorch's own fp16,
one `gelu` carried rel rms 8.94e-4 narrow and 2.48e-6 wide — 360x from the width
alone, same coefficients. End to end the encoder went 1.349e-1 -> 2.968e-2, and
at the last residual its distance from the *fp32* truth went from ~5x PyTorch's
own fp16 drift to 1.15x. The remaining gap is eight of 1500 frames where the
model is ill-conditioned in fp16 and PyTorch's own fp16 breaks down too (its
error at those frames jumps 12x in one block).

So: this is a correctness bug in an activation, found by an accuracy target on a
model, and it is worth a test of its own because the *next* half-precision export
will hit it in whatever op is widened next.

## The assertions

Against Float64 evaluated the same way, which is the thing both versions are
approximating. The thresholds are far from both measurements — the narrow gelu
was 8.9e-4 and the wide one is ~2e-6 — so this pins the behaviour rather than a
measurement of one machine. Host-only: it is the *element function*, and the
element function is the same expression the GPU epilogue compiles.
"""

using Test, DNNKernels
const DKA = DNNKernels

# A grid dense where it matters. gelu's cancellation lives at x < -2, and the
# fp16 spacing there is 2^-11 relative, so a random sample would mostly miss it.
const XS16 = sort(unique(vcat(Float16.(-8:0.01:8), Float16[-4, -3, -2, -1, 0, 1, 2, 3, 4])))

"""
Float64 reference for the exact gelu, on the same fp16 inputs.

The *same* A&S approximation, in Float64. That is the point: this file tests the
evaluation width, not the choice of approximation, so the reference has to differ
from the implementation in width alone. A&S 7.1.26's own 1.5e-7 error floor sits
three orders of magnitude below anything fp16 can represent, so it cannot mask
what is being measured.
"""
gelu64(x) = (t = Float64(x); 0.5t * (1 + erf64(t / sqrt(2.0))))
erf64(t) = (a = abs(t); s = t < 0 ? -1.0 : 1.0;
            u = 1 / (1 + 0.3275911a);
            s * (1 - (((((1.061405429u - 1.453152027)u) + 1.421413741)u -
                        0.284496736)u + 0.254829592) * u * exp(-a * a)))

"""Float64 reference for the tanh approximation, on the same fp16 inputs."""
gelutanh64(x) = (t = Float64(x);
                 0.5t * (1 + tanh(0.7978845608028654 * (t + 0.044715t^3))))

relrms(a, b) = sqrt(sum(abs2, Float64.(a) .- Float64.(b)) / sum(abs2, Float64.(b)))

@testset "activation width" begin
    @testset "erf at half precision" begin
        got = DKA.erf.(XS16)
        @test eltype(got) === Float16           # the op's dtype stays the graph's
        want = erf64.(Float64.(XS16))
        # Float16 can hold erf to ~5e-4 relative; anything above that is the
        # evaluation, not the storage.
        @test relrms(got, want) < 3e-4
        # The tail is where the narrow version returned zero for a nonzero answer.
        @test DKA.erf(Float16(-4)) < -0.999
        @test DKA.erf(Float16(4)) > 0.999
    end

    @testset "gelu, exact form" begin
        got = DKA.geluexact.(XS16)
        @test eltype(got) === Float16
        want = gelu64.(XS16)
        @test relrms(got, want) < 1e-4          # narrow measured 8.9e-4 on real data
        # gelu(-4) is -1.3e-4, small but not zero; the narrow form returned 0.
        @test DKA.geluexact(Float16(-4)) != 0
        @test abs(Float64(DKA.geluexact(Float16(-3))) - gelu64(Float16(-3))) < 1e-4
    end

    @testset "gelu, tanh form" begin
        got = DKA.gelutanh.(XS16)
        @test eltype(got) === Float16
        @test relrms(got, gelutanh64.(XS16)) < 1e-4
        # x^3 in half overflows above |x| = 40, which the residual stream reaches.
        @test isfinite(DKA.gelutanh(Float16(64)))
        @test DKA.gelutanh(Float16(64)) ≈ Float16(64)
        @test DKA.gelutanh(Float16(-64)) == 0
    end

    @testset "fp32 and fp64 operands are unchanged" begin
        xs = Float32.(-8:0.01:8)
        @test eltype(DKA.geluexact.(xs)) === Float32
        @test relrms(DKA.geluexact.(xs), gelu64.(xs)) < 1e-6
        @test DKA.geluexact(1.0) isa Float64
    end

    # The fused epilogue and the standalone op are the same function now, not two
    # copies of one expression — `runop!` dispatches to these. Folding a gelu into
    # a GEMM store must therefore be exactly a no-op on values, and this is what
    # says so without needing a device.
    @testset "the epilogue is the op" begin
        @test DKA.actfn(:gelu) === DKA.geluexact
        @test DKA.actfn(:relu)(Float16(-1)) === Float16(0)
        @test DKA.actfn(:none) === identity
    end
end
