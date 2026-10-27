using Test
# Test-only renderer: no candidate evaluation, process launch, or GPU access.
const mock_render_calls = Ref(0)
function render_julia_candidate(source, expression; artifact_path)
    mock_render_calls[] += 1
    @test expression == "render_plan()"
    @test occursin("SafePlotBuilder.build", source)
    (;rendered=true, layout=(;axes=[(;within_canvas=true)]))
end
include("render_plot_interface_audit.jl")
include("plot_interface_references.jl")

function test_render_plot_interface_audit()
    save(path, value)=open(io->JSON3.write(io,value),path,"w")
    digest(path)=bytes2hex(sha256(read(path)))
    @testset "Plot audit roundtrip and first-attempt accounting (mock renderer)" begin
        for reference in plot_interface_references()
            spec=SafePlotSpec.parse_spec(JSON3.write(reference))
            reparsed=SafePlotSpec.parse_spec(JSON3.write(plot_render_packet(spec)))
            @test plot_specs_equivalent(spec,reparsed)
        end
        mktempdir() do directory
            jobs=[(;id="mock_$(i)",arm="plot_spec") for i in 1:16]
            conditions=joinpath(directory,"conditions.json")
            save(conditions,(;jobs))
            registration=freeze_plot_render_audit(;directory)
            @test length(registration.jobs)==16
            @test_throws ErrorException freeze_plot_render_audit(;directory)
            @test_throws SystemError render_plot_interface_audit(;directory)
            rows=map(jobs) do job
                answer=JSON3.write(first(plot_interface_references()))
                output=joinpath(directory,job.id*".json")
                save(output,(;answer))
                (;id=job.id,generation_sha256=digest(output),admitted=job.id!="mock_16")
            end
            save(joinpath(directory,"summary.json"),(;complete=true,conditions_sha256=digest(conditions),rows))
            @test_throws ArgumentError render_plot_interface_audit(;directory,max_cases=true)
            @test_throws ArgumentError render_plot_interface_audit(;directory,max_cases=0)
            result=render_plot_interface_audit(;directory,max_cases=1)
            @test !result.complete
            @test count(r->r.status=="rendered_unclipped",result.rows)==1
            @test count(r->r.status=="pending",result.rows)==14
            @test result.rows[16].status=="unsupported_or_truncated_output"
            artifacts=joinpath(directory,"render_artifacts")
            save(joinpath(artifacts,"mock_2_attempt.json"),(;interrupted=true))
            result=render_plot_interface_audit(;directory,max_cases=16)
            @test !result.complete
            @test result.rows[2].status=="interrupted_first_render"
            @test count(r->r.status=="pending",result.rows)==0
            @test mock_render_calls[]==14
            repeated=render_plot_interface_audit(;directory,max_cases=16)
            @test !repeated.complete
            @test mock_render_calls[]==14 # cached first attempts NEVER regenerated
            @test all(r->r.aesthetic_score===nothing,repeated.rows)
            save(joinpath(directory,"mock_1.json"),(;answer="changed"))
            @test_throws ErrorException render_plot_interface_audit(;directory)
        end
    end
end
