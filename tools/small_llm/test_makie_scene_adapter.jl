include("makie_scene_adapter.jl")
using .MakieSceneAdapter, JSON3, Test, SHA
isdefined(@__MODULE__,:JuliaTaskEval) || include("julia_eval.jl")

# Trusted reference fixtures only. Never include these in model prompts.
const SCENE_FIXTURES=Dict(
    "makie_link"=>(inputs=["x","y"],spec=Dict(
        "size"=>[720,480],"axes"=>[
            Dict("id"=>"top","row"=>1,"col"=>1,"title"=>"Signal","ylabel"=>"Amplitude"),
            Dict("id"=>"bottom","row"=>2,"col"=>1,"xlabel"=>"Time (s)","ylabel"=>"Magnitude")],
        "plots"=>[Dict("kind"=>"lines","axis"=>"top","x"=>"x","y"=>"y"),
            Dict("kind"=>"scatter","axis"=>"bottom","x"=>"x","y"=>Dict("op"=>"abs","arg"=>"y"))],
        "links"=>[Dict("dimension"=>"x","axes"=>["top","bottom"])],"returns"=>["fig","top","bottom"])),
    "makie_band"=>(inputs=["x","mu","sigma"],spec=Dict(
        "size"=>[640,400],"axes"=>[Dict("id"=>"main","row"=>1,"col"=>1,"title"=>"Estimate","xlabel"=>"Time","ylabel"=>"Value")],
        "plots"=>[Dict("kind"=>"band","axis"=>"main","x"=>"x",
            "lower"=>Dict("op"=>"-","left"=>"mu","right"=>"sigma"),
            "upper"=>Dict("op"=>"+","left"=>"mu","right"=>"sigma"),"color"=>"blue","alpha"=>0.25),
            Dict("kind"=>"lines","axis"=>"main","x"=>"x","y"=>"mu","color"=>"blue","label"=>"Mean")],
        "legends"=>[Dict("axis"=>"main","position"=>"lt")],"returns"=>["fig","main"]))
)

function test_scene_adapter(;execute=true)
    loosen(x::AbstractDict)=Dict{String,Any}(String(k)=>loosen(v) for (k,v) in x)
    loosen(x::AbstractVector)=Any[loosen(v) for v in x]
    loosen(x)=x
    @testset "scene validation" begin
        original=SCENE_FIXTURES["makie_link"]
        fixture=(;original.inputs,spec=loosen(original.spec))
        compile(spec)=MakieSceneAdapter.compile_scene(JSON3.write(spec);inputs=fixture.inputs)
        @test Meta.parseall(compile(fixture.spec)) isa Expr
        for mutate in (
            s->s["arbitrary_code"]="run(`id`)",
            s->s["axes"][2]["id"]="top",
            s->s["axes"][1]["id"]="fig",
            s->s["size"][1]=100000,
            s->s["plots"][1]["y"]="read(secret)",
            s->s["plots"][1]["y"]=Dict("op"=>"eval","arg"=>"y"),
            s->s["plots"][1]["alpha"]=2,
            s->s["plots"][1]["kind"]="eval",
            s->s["links"][1]["dimension"]="z",
            s->s["returns"]=["secret"])
            s=deepcopy(fixture.spec);mutate(s)
            @test_throws ArgumentError compile(s)
        end
        # Labels are encoded as Julia strings, not spliced as source.
        s=deepcopy(fixture.spec);s["axes"][1]["title"]="\"); error(\"injected\"); #"
        code=compile(s)
        @test occursin(repr(s["axes"][1]["title"]),code)
        @test Meta.parseall(code) isa Expr
        @test_throws ArgumentError MakieSceneAdapter.compile_scene("x"^16385;inputs=fixture.inputs)
        expr="y"
        for _ in 1:6;expr=Dict("op"=>"abs","arg"=>expr);end
        @test_throws ArgumentError MakieSceneAdapter.dataexpr(expr,fixture.inputs)
    end
    rows=Any[]
    if execute
        for task in JuliaTaskEval.JULIA_TASKS
            haskey(SCENE_FIXTURES,task.id) || continue
            fixture=SCENE_FIXTURES[task.id];raw=JSON3.write(fixture.spec)
            code=MakieSceneAdapter.compile_scene(raw;inputs=fixture.inputs)
            result=JuliaTaskEval.evaluate(task,code)
            push!(rows,(;result...,contract_sha256=bytes2hex(sha256(task.tests))))
            @test result.passed
        end
    end
    report=(;scope="trusted adapter qualification only; no model output or accuracy improvement",execution=rows,
        adapter_sha256=bytes2hex(sha256(read(joinpath(@__DIR__,"makie_scene_adapter.jl")))))
    open(io->JSON3.pretty(io,report),joinpath(@__DIR__,"makie_scene_adapter_validation.json"),"w")
    report
end
