include("plot_interface_tasks.jl")
include("plot_interface_references.jl")
include("plot_interface_admission.jl")
using Test,JSON3,SHA
function qualify_plot_interface_references(;destination=joinpath(@__DIR__,"plot_interface_reference_validation.json"))
    isfile(destination)&&error("Preserve qualification")
    tasks=plot_interface_tasks();references=plot_interface_references();rows=Any[]
    tests=@testset "New8 reference interface plans and private varied data" begin
        @test length(tasks)==length(references)==8
        @test length(unique(t.id for t in tasks))==8
        for (task,reference) in zip(tasks,references)
            spec=SafePlotSpec.parse_spec(JSON3.write(reference))
            native=plot_interface_native_reference(spec)
            lowered=admit_plot_interface(native,"julia_source")
            declarative=admit_plot_interface(JSON3.write(reference),"plot_spec")
            @test lowered.accepted
            @test declarative.accepted
            @test plot_specs_equivalent(lowered.spec,spec)
            @test plot_specs_equivalent(declarative.spec,spec)
            for n in (17,64)
                x=collect(range(0.1,3.0;length=n));a=2 .+sin.(x);b=0.5 .+exp.(-x./3)
                data=Dict("x"=>x,"a"=>a,"b"=>b,"lo"=>a.-0.15,"hi"=>a.+0.15)
                @test SafePlotSpec.validate_data(spec,data)===nothing
            end
            wrong=JSON3.read(JSON3.write(reference),Dict{String,Any})
            wrong["series"][1]["color"]=reference.series[1].color=="blue" ? "red" : "blue"
            @test !plot_specs_equivalent(SafePlotSpec.parse_spec(JSON3.write(wrong)),spec)
            push!(rows,(;id=task.id,reference_sha256=bytes2hex(sha256(JSON3.write(reference))),
                native_reference_sha256=bytes2hex(sha256(native)),correct_reference=true,wrong_color_rejected=true))
        end
        for arm in ("julia_source","plot_spec")
            code=arm=="julia_source" ? plot_interface_native_reference(SafePlotSpec.parse_spec(JSON3.write(first(references)))) : JSON3.write(first(references))
            language=arm=="julia_source" ? "julia" : "json"
            @test plot_interface_payload("```"*language*"\n"*code*"\n```",arm)==strip(code)
            @test !admit_plot_interface("prefix\n```"*language*"\n"*code*"\n```",arm).accepted
            @test !admit_plot_interface("```bad\n"*code*"\n```",arm).accepted
        end
        # An ineffective legend flag must not manufacture a failure when all
        # series have empty labels and neither builder creates a legend.
        nolegend=references[7]
        @test plot_specs_equivalent(SafePlotSpec.parse_spec(JSON3.write(nolegend)),SafePlotSpec.parse_spec(JSON3.write((;nolegend...,legend=true))))
    end
    count=Test.get_test_counts(tests)
    result=(;complete=true,checks=count.passes+count.cumulative_passes,rows,
        scope="Private reference/frontend/data/mutant qualification only; no model plot answers or aesthetic score. Restricted Julia is compiled into the SAME trusted drawing plan, never executed.",
        sources=Dict(file=>bytes2hex(sha256(read(joinpath(@__DIR__,file)))) for file in
            ("plot_interface_tasks.jl","plot_interface_references.jl","plot_interface_admission.jl","safe_plot_spec.jl","safe_plot_builder.jl","safe_plot_julia_source.jl","qualify_plot_interface_references.jl")))
    open(io->JSON3.pretty(io,result),destination,"w");result
end
