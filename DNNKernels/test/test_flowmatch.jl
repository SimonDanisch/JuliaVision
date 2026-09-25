# The shared flow-matching sampler, pinned bit for bit against diffusers.
#
# Every expected value here is a checksum of what diffusers' own
# `FlowMatchEulerDiscreteScheduler` (and, for guidance, PyTorch) produced from the
# same inputs, at the `diffusers-src` commit in `tools/Artifacts.toml`. The inputs
# are fp16 bit patterns built from integers so both sides construct them exactly.
# `tools/flowmatch_checksums.py` regenerates every number here.
#
# Two copies this replaced were wrong in ways only this kind of test sees: Qwen's
# schedule differed from diffusers in the last bit for 99 of 300 image sizes and
# step counts, and Hunyuan3D's guidance rounded the scale to fp16 before the
# multiply. The GPU half matters as much as the host half: Lava keeps a fused fp16
# broadcast in fp32, so folding the step's two broadcasts into one passes on the
# CPU and fails there.

using Test, Lava, DNNKernels, KernelAbstractions
import Mantle
using Mantle: LavaBackend
using DNNKernels: linspace, flowschedule, NoShift, StaticShift, ExponentialShift,
                  calculate_shift, eulerstep, eulerstep!, cfg

const FLOW_N = 4096

# The same construction as the Python side: sign, exponent and mantissa from integers.
fp16input(mul, e0) = collect(reinterpret(Float16,
    UInt16[(((k * 40503) >> 5) & 1) << 15 | (e0 + (k * 7) % 8) << 10 | (k * mul) % 1024
           for k in 0:(FLOW_N - 1)]))

# Position-weighted sum of the bit patterns, mod 2^64.
bitsum(a::AbstractVector{Float16}) = sum(UInt64(k) * UInt64(b) for (k, b) in enumerate(reinterpret(UInt16, a)))
bitsum(a::AbstractVector{Float32}) = sum(UInt64(k) * UInt64(b) for (k, b) in enumerate(reinterpret(UInt32, a)))

const X = fp16input(2654435761, 11)
const V = fp16input(2246822519, 12)
const C = fp16input(3266489917, 10)
const U = fp16input(668265263, 10)

@testset "flow matching: inputs are the ones diffusers saw" begin
    @test bitsum(X) == 266406547456
    @test bitsum(V) == 275002138624
    @test bitsum(C) == 257818562560
    @test bitsum(U) == 257808048128
end

@testset "flow matching: Qwen-Image 2.1's schedule" begin
    # `scheduler_config.json`: exponential dynamic shift, terminal stretched to 0.02.
    qwen(n, steps) = flowschedule(linspace(1.0, 1 / steps, steps),
                                  ExponentialShift(calculate_shift(n; max_seq_len = 8192, max_shift = 0.9));
                                  shiftterminal = 0.02)
    s = qwen(64 * 64, 25)
    @test reinterpret(UInt32, s.sigmas) == UInt32[
        0x3f800000, 0x3f7a7491, 0x3f74adf8, 0x3f6ea86d, 0x3f685fd2, 0x3f61cfb3, 0x3f5af32b,
        0x3f53c4e8, 0x3f4c3f10, 0x3f445b39, 0x3f3c125a, 0x3f335ca8, 0x3f2a318a, 0x3f20877f,
        0x3f1653ed, 0x3f0b8b09, 0x3f001fa6, 0x3ee805f0, 0x3ece48c2, 0x3eb2e232, 0x3e95a7b0,
        0x3e6cd194, 0x3e29dc40, 0x3dc3e088, 0x3ca3d700, 0x00000000]
    @test reinterpret(UInt32, s.timesteps) == UInt32[
        0x447a0000, 0x447495d6, 0x446ef1e4, 0x4469107a, 0x4462ed93, 0x445c84d5, 0x4455d178,
        0x444ece4b, 0x44477596, 0x443fc116, 0x4437a9ec, 0x442f287c, 0x44263461, 0x441cc452,
        0x4412cdf5, 0x440845c7, 0x43fa3dd0, 0x43e295cc, 0x43c9730d, 0x43aeb0e5, 0x439225c2,
        0x436744ab, 0x4325e116, 0x42bf4945, 0x419ffff6]
    # 472 schedules, 2 to 60 steps at eight token counts: 1024x1024, 1408x1056,
    # 1664x928 and 1328x1328 among them. The copy this replaced computed `mu` in
    # Float32, which is exact at round token counts and passed the case above;
    # at 1664x928 it was wrong for 57 of the 59 step counts.
    runs = [qwen(n, steps) for n in (256, 1024, 4096, 5808, 6032, 6889, 494, 1665) for steps in 2:60]
    @test bitsum(reduce(vcat, [r.sigmas for r in runs])) == 116855988325820340
    @test bitsum(reduce(vcat, [r.timesteps for r in runs])) == 122019502778554299
end

@testset "flow matching: a static shift" begin
    runs = [flowschedule(linspace(1.0, 0.0, steps + 1)[1:steps], StaticShift(5.0)).sigmas for steps in 2:60]
    @test bitsum(reduce(vcat, runs)) == 1849263390641458
end

@testset "flow matching: linspace rounds like numpy" begin
    @test linspace(1.0, 0.0, 1) == [1.0]
    @test linspace(0.0, 1.0, 5) == [0.0, 0.25, 0.5, 0.75, 1.0]
    # numpy's values, where the last one is also where Julia's `range` differs.
    @test linspace(1.0, 1 / 30, 30)[[12, 18, 24]] ==
          [0x1.4444444444444p-1, 0x1.bbbbbbbbbbbbcp-2, 0x1.ddddddddddde0p-3]
end

# The step and guidance on host arrays and on the device, against the same sums.
function flowchecks(todev, back)
    @testset "Euler step, fp16, 0.8 -> 0.5" begin
        @test bitsum(back(eulerstep(todev(X), todev(V), 0.8f0, 0.5f0))) == 250442992316
        x = todev(copy(X))
        eulerstep!(x, todev(V), 0.8f0, 0.5f0)
        @test bitsum(back(x)) == 250442992316
    end
    @testset "Euler step, fp16, a step size fp16 cannot hold" begin
        @test bitsum(back(eulerstep(todev(X), todev(V), 0f0, Float32(1 / 49)))) == 266924126344
    end
    @testset "Euler step, fp32" begin
        @test bitsum(back(eulerstep(todev(Float32.(X)), todev(Float32.(V)), 0.8f0, 0.5f0))) ==
              17715175474257672
    end
    @testset "guidance at scale $scale" for (scale, want) in
            ((5.0, 256781869600), (3.3, 254925809232), (7.5, 261275841672))
        @test bitsum(back(cfg(todev(C), todev(U), scale))) == want
    end
end

# The step unwraps `Mantle.Buffer`s and re-dispatches; two arrays of different
# element types used to take that path again and overflow the stack.
@testset "flow matching: a step over two dtypes is refused" begin
    @test_throws ArgumentError eulerstep(Float16[1], Float32[1], 0f0, 1f0)
    @test_throws ArgumentError eulerstep!(Float16[1], Float32[1], 0f0, 1f0)
end

@testset "flow matching: host" begin
    flowchecks(identity, identity)
end

@testset "flow matching: device" begin
    dev = LavaBackend()
    function todev(a)
        d = KernelAbstractions.allocate(dev, eltype(a), size(a))
        copyto!(d, a)
        return d
    end
    flowchecks(todev, Array)
end

# What the runners actually hold: `toback` returns a `Mantle.Buffer`, not an array.
@testset "flow matching: a Mantle.Buffer, as toback returns it" begin
    dev = LavaBackend()
    xb, vb = DNNKernels.toback(dev, copy(X)), DNNKernels.toback(dev, V)
    @test xb isa Mantle.Buffer
    @test bitsum(Array(eulerstep(xb, vb, 0.8f0, 0.5f0))) == 250442992316
    @test eulerstep!(xb, vb, 0.8f0, 0.5f0) === xb
    @test bitsum(Array(Mantle.storage(xb))) == 250442992316
end
