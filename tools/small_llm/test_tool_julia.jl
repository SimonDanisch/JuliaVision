# The tools a kernel answer may call, and the loop that serves them. The loop's
# rewind is exact or it is wrong: with the same seed, 16-token chunks (which run
# past a call and are rewound) must give the same calls, tokens and answer as
# 1-token chunks (which stop exactly at every call and never rewind).
isdefined(@__MODULE__,:tool_julia_answer) || include("tool_julia.jl")
using Test,JSON3

function test_julia_tools()
    J=JuliaTools
    @testset "check_syntax and lookup" begin
        @test J.check_syntax("function kernel!(x)\n    x[1]=1\n    nothing\nend")=="no syntax errors"
        bad=J.check_syntax("using KernelInterface as KI\nfunction f() end")
        @test occursin("ParseError",bad) && occursin("as",bad)
        @test occursin("ParseError",J.check_syntax("function f(\n"))
        missing=J.lookup("Atomix.atomic_max")
        @test occursin("`Atomix.atomic_max` is not defined",missing)
        @test occursin("@atomic",missing) && occursin("modify!",missing)
        @test occursin("superset of `Base.@atomic`",J.lookup("Atomix.@atomic"))
        @test occursin("Methods:",J.lookup("`KI.get_global_id()`"))
        @test occursin("max",J.lookup("Base.@atomic"))
        @test occursin("exports",J.lookup("Atomix"))
        @test occursin("not one of the modules",J.lookup("Foo.bar"))
        @test occursin("not a module",J.lookup("KI.get_global_id.x"))
        @test length(J.lookup("Base"))<=J.MAXCHARS+20
        box=J.parse_lookup()
        @test startswith(J.respond(box,"expected <function=NAME>"),"error: malformed tool call")
        @test startswith(J.respond(box,QwenTools.Call("run",["code"=>"1"])),"error: unknown function")
        @test startswith(J.respond(box,QwenTools.Call("lookup",Pair{String,String}[])),"error: missing parameter")
        @test J.respond(box,QwenTools.Call("lookup",["name"=>"Atomix.atomic_max"]))==missing
        @test occursin("available: none",J.respond(J.notools(),QwenTools.Call("lookup",["name"=>"x"])))
    end
end

function test_run_tool()
    box=parse_lookup_run()
    try
        @testset "run_julia through the toolbox" begin
            @test "run_julia" in [t.name for t in box.tools]
            k="function kernel!(out,x,n::Int32)\n    i=KI.get_global_id().x\n    i<=n && (@inbounds out[i]=x[i]+1f0)\nend\n"*
              "__bench_launch__(kernel!,zeros(Float32,3),Float32[1,2,3],Int32(3);threads=3)"
            r=JuliaTools.respond(box,QwenTools.Call("run_julia",["code"=>k]))
            # The kernel returns a value: the compiler says so, which is the point.
            @test occursin("Error:",r) && occursin("return",r)
            r=JuliaTools.respond(box,QwenTools.Call("run_julia",["code"=>replace(k,"+1f0)\nend"=>"+1f0)\n    nothing\nend")]))
            @test r=="Value: Float32[2.0, 3.0, 4.0]"
        end
    finally
        close(box)
    end
end

function test_tool_loop(model,tokenizer,task;seed=20261024,max_tokens=3000,think_budget=nothing,want=nothing)
    messages=[QwenTools.Message("system",TOOL_SYSTEM),QwenTools.Message("user",task.prompt)]
    a=tool_julia_answer(model,tokenizer,messages;max_tokens,seed,chunk=16,think_budget)
    b=tool_julia_answer(model,tokenizer,messages;max_tokens,seed,chunk=1,think_budget)
    @testset "tool loop rewinds exactly (budget $think_budget)" begin
        want===nothing || @test (a.generated_tokens,length(a.calls))==want
        think_budget===nothing || @test a.forced_closes>=1 && a.forced_closes==b.forced_closes
        # Something must have stopped mid-chunk, or the rewind went untested.
        @test !isempty(a.calls) || a.forced_closes>0
        @test a.rewinds>0
        @test b.rewinds==0
        @test a.thinking_tokens==b.thinking_tokens
        @test a.calls==b.calls
        @test a.answer==b.answer
        @test (a.generated_tokens,a.context_tokens)==(b.generated_tokens,b.context_tokens)
    end
    a
end

# Drafts change how many forward passes a round takes, never what it samples:
# with the same seed, speculation must give the same calls, tokens and answer,
# with drafts from the context (`speculate=true`) or the MTP layer (`:mtp`).
function test_tool_loop_speculative(model,tokenizer,task;seed=20261024,max_tokens=3000,think_budget=nothing,speculate=true)
    messages=[QwenTools.Message("system",TOOL_SYSTEM),QwenTools.Message("user",task.prompt)]
    a=tool_julia_answer(model,tokenizer,messages;max_tokens,seed,chunk=16,think_budget)
    b=tool_julia_answer(model,tokenizer,messages;max_tokens,seed,chunk=16,think_budget,speculate)
    @testset "tool loop with $(speculate===:mtp ? "MTP" : "context") drafts samples what it samples without" begin
        @test a.thinking_tokens==b.thinking_tokens
        @test a.calls==b.calls
        @test a.answer==b.answer
        @test (a.generated_tokens,a.context_tokens,a.rewinds)==(b.generated_tokens,b.context_tokens,b.rewinds)
    end
    (;plain=a,speculative=b)
end

# A frozen tool-loop receipt with calls (so rewinds after calls happen) run again
# with and without drafts: both reproduce its answer and calls.
function test_receipt_speculative(model,tokenizer;path=joinpath(@__DIR__,"kernel_tools_v2_results","ki_body_atomic_max_s20261025_tools_budget.json"),
                                  max_tokens=2000,think_budget=3072,speculate=true)
    r=JSON3.read(read(path,String))
    messages=[QwenTools.Message("system",String(r.system)),QwenTools.Message("user",String(r.prompt))]
    run(speculate)=tool_julia_answer(model,tokenizer,messages;max_tokens,seed=UInt32(r.seed),think_budget,speculate)
    a=run(false);b=run(speculate)
    @testset "frozen receipt, with and without $(speculate===:mtp ? "MTP" : "context") drafts" begin
        @test a.answer==String(r.answer)
        @test length(a.calls)==length(r.calls)>0
        @test a.rewinds>0
        @test b.answer==a.answer
        @test b.calls==a.calls
        @test (b.generated_tokens,b.context_tokens,b.rewinds,b.thinking_tokens)==(a.generated_tokens,a.context_tokens,a.rewinds,a.thinking_tokens)
    end
    (;plain=a,speculative=b)
end

# A prefix kept from one session and restored into the loop's own: the same
# answer, calls and tokens as ingesting the whole prompt. Bit-exact chunks may
# cut anywhere; a wide chunk's prefix ends on a chunk boundary (the loop refuses
# one that does not). With `speculate=:mtp` the prefix is kept from an MTP
# speculator, its layer's cache with it, and the loop drafts with the layer.
function test_prefix_cache(model,tokenizer;path=joinpath(@__DIR__,"kernel_tools_v2_results","ki_body_atomic_max_s20261025_tools_budget.json"),
                           max_tokens=600,session=(;),prompt_chunk=INGEST_CHUNK[],keep=40,speculate=false)
    r=JSON3.read(read(path,String));T=SmallLLMChecks.QwenText
    messages=[QwenTools.Message("system",String(r.system)),QwenTools.Message("user",String(r.prompt))]
    toolbox=JuliaTools.parse_lookup()
    ids=try
        T.encode(tokenizer,QwenTools.render(messages;tools=toolbox.tools,thinking=true))
    finally
        close(toolbox)
    end
    head=prompt_chunk<=INGEST_CHUNK[] ? ids[1:end-keep] : ids[1:prompt_chunk*fld(length(ids)-keep,prompt_chunk)]
    isempty(head) && throw(ArgumentError("the prompt is shorter than one $prompt_chunk-token chunk"))
    s=textsession(model;capacity=length(head)+16,session...);cache=nothing;sp=nothing
    try
        sp=speculate===:mtp ? mtpspeculator(s) : nothing;c=consumer(s,sp)
        ingest!(c,head;chunk=prompt_chunk);Mantle.waitidle(model.device)
        cache=PrefixCache(c,head)
    finally
        sp===nothing||close(sp);close(s)
    end
    try
        run(prefix)=tool_julia_answer(model,tokenizer,messages;max_tokens,seed=UInt32(r.seed),think_budget=3072,session,prompt_chunk,prefix,speculate)
        if speculate===:mtp
            @testset "an MTP prefix needs an MTP speculator" begin
                @test_throws ArgumentError tool_julia_answer(model,tokenizer,messages;max_tokens=1,seed=UInt32(r.seed),session,prompt_chunk,prefix=cache)
            end
        end
        a=run(nothing);b=run(cache)
        @testset "prompt prefix restored instead of ingested" begin
            if prompt_chunk>INGEST_CHUNK[]
                # One token short of the boundary: refused before anything runs.
                @test_throws ArgumentError tool_julia_answer(model,tokenizer,messages;max_tokens=1,seed=UInt32(r.seed),session,prompt_chunk,
                    prefix=PrefixCache(cache.tokens[1:end-1],cache.kv,cache.recurrent))
            end
            @test b.answer==a.answer
            @test b.calls==a.calls
            @test (b.generated_tokens,b.context_tokens,b.thinking_tokens)==(a.generated_tokens,a.context_tokens,a.thinking_tokens)
        end
        (;full=a,restored=b)
    finally
        close(cache)
    end
end
