include("laptop_shared_source.jl")
include("sampled_example_rpn_program.jl")
using Test,JSON3,SHA

function freeze_laptop_example_identity_gate(;directory=joinpath(@__DIR__,"laptop_example_identity_gate"))
    ispath(directory)&&!isempty(readdir(directory))&&error("Preserve identity registration")
    parent=joinpath(@__DIR__,"laptop_example_constraint_gate/result.json")
    receipt=JSON3.read(read(parent,String));receipt.complete&&receipt.normal_final_waitidle&&receipt.passed==1627||error("Hardware gate incomplete")
    original=JSON3.read(read(joinpath(@__DIR__,"laptop_rpn_gate/conditions.json"),String))
    kernels=JSON3.read(read(joinpath(@__DIR__,"laptop_example_constraint_gate/conditions.json"),String))
    sources=merge(Dict(String(k)=>String(v) for (k,v) in pairs(original.source_sha256)),
        Dict(String(k)=>String(v) for (k,v) in pairs(kernels.sources)),
        Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in
            ("sampled_example_rpn_program.jl","laptop_example_identity_gate.jl")))
    all(bytes2hex(sha256(read(joinpath(@__DIR__,f))))==hash for (f,hash) in sources)||error("Source drift")
    p=(;scope="Full-model disabled numeric-hook identity ONLY, not a task-quality attempt or trained reasoning module",
        kernel_receipt_sha256=bytes2hex(sha256(read(parent))),kernel_pid=receipt.pid,
        portable_manifest_sha256=original.portable_manifest_sha256,sources,
        messages=["system"=>"Return only a postfix program, one action per line. Start with an empty stack. Actions: x0, x_prev_periodic, x_next_periodic, x_prev_clamped, x_next_clamped, zero, one, two, half, neg, square, add, sub, mul, min, max, finish. Constants push values; binary operations pop right then left. Finish with one value.",
            "user"=>"Return constant 0.5 for every input element. Public example: input [2.0], output [0.5]. No explanation or markdown."],
        public_cases=[(;input=Float32[2],output=Float32[0.5])],
        decoding=(;max_tokens=512,operation_budget=8,max_depth=4,seed=20261024,topk=20,
            temperature=0.6f0,top_p=0.95f0,presence=0f0,output_chunk=64))
    mkpath(directory);open(io->JSON3.pretty(io,p),joinpath(directory,"conditions.json"),"w");p
end

function run_laptop_example_identity_gate(;directory=joinpath(@__DIR__,"laptop_example_identity_gate"))
    normpath(pwd())=="/tmp/small-llm-bosgame-20261025"&&Sys.total_memory()<=40*1024^3||error("Laptop snapshot only")
    path=joinpath(directory,"conditions.json");p=JSON3.read(read(path,String))
    p.kernel_pid!=getpid()||error("Retire hardware gate worker first")
    for (file,hash) in pairs(p.sources)
        basename(String(file))==String(file)&&bytes2hex(sha256(read(joinpath(@__DIR__,String(file)))))==hash||error("Source drift")
    end
    bytes2hex(sha256(read("gen/qwen35-4b-portable/portable.json")))==p.portable_manifest_sha256||error("Model drift")
    output=joinpath(directory,"result.json");ispath(output)&&error("Preserve identity attempt")
    report=Dict{String,Any}("complete"=>false,"pid"=>getpid(),"conditions_sha256"=>bytes2hex(sha256(read(path))),"scope"=>p.scope)
    save()=open(io->JSON3.pretty(io,report),output,"w")
    let
        let
            save();device=Mantle.Device("radeon");model=nothing
            try
                model=load_portable_qwen35(SmallLLMChecks.Qwen35Experiment,"gen/qwen35-4b-portable",device)
                tokenizer=SmallLLMChecks.QwenText.Tokenizer("gen/qwen35-4b-portable")
                messages=[String(first(keys(m)))=>String(first(values(m))) for m in p.messages]
                d=p.decoding;options=(;max_tokens=Int(d.max_tokens),operation_budget=Int(d.operation_budget),seed=UInt32(d.seed),
                    topk=Int(d.topk),temperature=Float32(d.temperature),top_p=Float32(d.top_p),presence=Float32(d.presence),output_chunk=Int(d.output_chunk))
                bank=RPNExampleSemantics.Bank([(;input=Float32.(c.input),output=Float32.(c.output)) for c in p.public_cases];max_depth=Int(d.max_depth))
                original=sampled_rpn_program(model,tokenizer,messages;enabled=true,max_depth=Int(d.max_depth),options...)
                report["original"]=original;save()
                disabled=sampled_example_rpn_program(model,tokenizer,messages,bank;enabled=false,options...)
                report["disabled"]=disabled;save()
                tests=@testset "Real4B disabled public-example hook identity, not quality" begin
                    for field in (:answer,:code,:input_tokens,:generated_tokens,:decode_steps_executed,:decode_submissions,:truncated,:input_sha256)
                        @test getproperty(original,field)==getproperty(disabled,field)
                    end
                    @test original.sampling.draws==disabled.sampling.draws
                    @test disabled.constraint.raw_status==disabled.constraint.sampler_status==disabled.constraint.rejected==0
                    @test disabled.constraint.accepted && disabled.constraint.done==1
                    @test original.grammar.operations==disabled.constraint.operations
                end
                c=Test.get_test_counts(tests);report["passed"]=c.passes+c.cumulative_passes;save()
            finally
                try model===nothing||close(model)
                finally finish_device!(device) end
            end
            report["normal_final_waitidle"]=true;report["complete"]=true;save();report
        end
    end
end
