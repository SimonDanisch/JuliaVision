using Test, PPOCRRunner, DNNKernels, ColorTypes
import KernelAbstractions as KA

# What the exporter leaves beside the graphs: PyTorch's answers on the inputs it
# exported with. Not in the artifacts (the people using the runner do not need
# them), so these checks run where the exports were made and are skipped
# elsewhere.
const GEN = let dir = normpath(joinpath(@__DIR__, "..", "..", "gen", "graphs"))
    isdir(dir) ? dir : normpath(joinpath(@__DIR__, "..", "..", "..", "..", "gen", "graphs"))
end
reference(name) = (p = joinpath(GEN, name, "reference.safetensors"); isfile(p) ? readsafetensors(p) : nothing)

@testset "PPOCRRunner" begin
    if !PPOCRRunner.ready()
        @info "PP-OCRv6 exports not installed; nothing to test"
        return
    end
    m = ppocr()

    @testset "against PyTorch" begin
        ref = reference("ppocrv6_det")
        if ref !== nothing
            copyto!(m.detinput, ref["input"])
            out = Array(first(DNNKernels.call(m.det, "ppocrv6_det", m.detinput; dims = (;))))
            @test maximum(abs.(out .- ref["probability"])) < 5e-3
        end
        ref = reference("ppocr_docori")
        if ref !== nothing
            copyto!(m.oriinput, ref["input"])
            out = vec(Array(first(DNNKernels.call(m.ori, "ppocr_docori", m.oriinput; dims = (;)))))
            @test maximum(abs.(out .- vec(ref["logits"]))) < 1e-3
        end
        ref = reference("ppocrv6_rec")
        if ref !== nothing
            x = ref["input"]
            input = KA.allocate(m.backend, Float32, size(x)...)
            copyto!(input, x)
            best, index = DNNKernels.call(m.rec, "ppocrv6_rec", input; dims = (w = size(x, 1) ÷ 16,))
            # The index per time step, not just the best value: `max.dim` once
            # returned the right maxima at the wrong positions.
            @test Array(index) == ref["index"]
        end
    end

    @testset "an empty page" begin
        page = fill(RGB{Float32}(0.95f0, 0.95f0, 0.95f0), 620, 877)
        @test isempty(detect(m, page))
        @test isempty(read_lines(m, page))
        @test PPOCRRunner.orientation(m, page) in (0, 90, 180, 270)
    end

    @testset "line widths are the planned ones" begin
        for (h, w) in ((40.0f0, 100.0f0), (40.0f0, 1500.0f0), (20.0f0, 4000.0f0))
            input, rw = PPOCRRunner.line_input(fill(RGB{Float32}(1, 1, 1), 64, 64), (0.0f0, 0.0f0, w, h))
            @test size(input, 1) in PPOCRRunner.REC_WIDTHS
            @test rw <= size(input, 1)
        end
    end
end
