isdefined(@__MODULE__,:render_julia_candidate) || include("julia_render.jl")
isdefined(@__MODULE__,:JuliaTaskEval) || include("julia_eval.jl")
using Test
function test_julia_render()
    @testset "bounded render artifacts" begin
        mktempdir() do directory
            file=joinpath(directory,"fake.png")
            @test_throws ArgumentError bounded_png_dimensions(file)
            write(file,zeros(UInt8,33))
            @test_throws ArgumentError bounded_png_dimensions(file)
            header=vcat(UInt8[0x89,0x50,0x4e,0x47,0x0d,0x0a,0x1a,0x0a,0,0,0,13],collect(codeunits("IHDR")),
                UInt8[0,0,0,1,0,0,0,1],zeros(UInt8,9))
            write(file,header)
            @test bounded_png_dimensions(file)==(;width=1,height=1)
            header[17:20]=UInt8[0,0,7,0xd1];write(file,header)
            @test_throws ArgumentError bounded_png_dimensions(file)
            header[17:20].=0;write(file,header)
            @test_throws ArgumentError bounded_png_dimensions(file)
            link=joinpath(directory,"link.png");symlink(file,link)
            @test_throws ArgumentError bounded_png_dimensions(link)
        end
    end
    task=only(filter(t->t.id=="makie_link",JuliaTaskEval.JULIA_TASKS))
    image=joinpath(@__DIR__,"render_results","trusted_makie_link.png")
    # These are public visualization inputs, distinct from contract values.
    expression="first(makeplot(collect(range(0,2pi;length=80)),sin.(collect(range(0,2pi;length=80)))))"
    result=render_julia_candidate(task.reference,expression;artifact_path=image)
    @test result.rendered
    @test result.dimensions==(;width=720,height=480)
    result
end

function render_retained_scene_candidates()
    expression="first(makeplot(collect(range(0,2pi;length=80)),sin.(collect(range(0,2pi;length=80)))))"
    paths=("julia_scene_v2_results/makie_link_baseline.json",)
    rows=Any[]
    for relative in paths
        path=joinpath(@__DIR__,relative)
        isfile(path) || error("retained candidate missing: $relative")
        candidate=JSON3.read(read(path,String))
        image=joinpath(@__DIR__,"render_results",basename(dirname(path))*"_makie_link.png")
        push!(rows,render_julia_candidate(String(candidate.code),expression;artifact_path=image))
    end
    rows
end
