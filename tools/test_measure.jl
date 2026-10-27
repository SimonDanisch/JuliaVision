# measure.jl loads on every machine; only a reading needs the clock. It used to
# resolve its telemetry source at include time, so on macOS every module that
# bundles a benchmark with its model (SmallLLMChecks, through bench_qwen35.jl)
# failed to load with an IOError from readdir("/sys/class/drm").
using Test

module MeasureLoad
include(joinpath(@__DIR__, "measure.jl"))
end

@testset "measure.jl loads without a clock, refuses only when it reads" begin
    @test MeasureLoad.TELEMETRY isa Base.OncePerProcess
    source = try
        MeasureLoad.telemetry()
    catch err
        err
    end
    if source isa Exception
        # The documented refusal, not a filesystem error.
        @test source isa ErrorException && occursin("publishes neither", source.msg)
        @test_throws ErrorException MeasureLoad.gpustate()
    else
        @test source isa Union{MeasureLoad.NvidiaSmi,MeasureLoad.AmdSysfs}
        state = MeasureLoad.gpustate()
        @test state.sm > 0 && state.smmax >= state.sm
    end
end
