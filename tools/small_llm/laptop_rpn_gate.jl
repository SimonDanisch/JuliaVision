include("laptop_shared_source.jl")
include("sampled_rpn_program.jl")
isdefined(@__MODULE__,:sampled_julia_answer)||include("sampled_julia.jl")
include("test_rpn_prefix_grammar_gpu.jl")
include("test_rpn_native_tokens_gpu.jl")
include("test_rpn_disabled_sampler_identity.jl")
using Test
laptop_rpn_digest(path)=bytes2hex(open(sha256,path))

function freeze_laptop_rpn_gate(;directory=joinpath(@__DIR__,"laptop_rpn_gate"),
        tokenizer_dir=abspath(joinpath(@__DIR__,"../../../../gen/qwen35-4b")))
    ispath(directory)&&!isempty(readdir(directory))&&error("preserve gate registration")
    proofs=Dict{String,String}()
    for file in ("rpn_host_validation.json","rpn_constraints_software_validation_v2.json","rpn_disabled_sampler_software_validation.json",
                 "rpn_pointwise_semantics_validation.json","sampled_rpn_options_validation.json")
        p=JSON3.read(read(joinpath(@__DIR__,file),String));p.complete||error("unqualified proof: $file")
        for (source,hash) in pairs(p.source_sha256)
            basename(String(source))==String(source)&&laptop_rpn_digest(joinpath(@__DIR__,String(source)))==hash||error("proof source drift: $source")
        end
        proofs[file]=laptop_rpn_digest(joinpath(@__DIR__,file))
    end
    host=JSON3.read(read(joinpath(@__DIR__,"rpn_host_validation.json"),String))
    host.grammar_checks==7700&&host.interface_checks==44||error("host checks")
    software=JSON3.read(read(joinpath(@__DIR__,"rpn_constraints_software_validation_v2.json"),String))
    software.grammar_device_checks==174&&software.native_tokens_device_checks==63||error("device checks")
    disabled=JSON3.read(read(joinpath(@__DIR__,"rpn_disabled_sampler_software_validation.json"),String))
    disabled.passed==320||error("disabled sampler checks")
    semantics=JSON3.read(read(joinpath(@__DIR__,"rpn_pointwise_semantics_validation.json"),String))
    length(semantics.rows)==5&&all(r->r.correct.passed&&!r.incorrect.passed,semantics.rows)||error("semantic checks")
    options=JSON3.read(read(joinpath(@__DIR__,"sampled_rpn_options_validation.json"),String))
    options.passed==10||error("option checks")
    base=laptop_shared_source_state(joinpath(@__DIR__,"bigcodebench_x_doc_results")).packet
    tokenizer=SmallLLMChecks.QwenText.Tokenizer(tokenizer_dir)
    grammar=RPNPrefixGrammar.Grammar(s->SmallLLMChecks.QwenText.encode(tokenizer,s),248320,Int(tokenizer.bpe.eos))
    extras=("laptop_rpn_gate.jl","sampled_rpn_program.jl","sampled_julia.jl","rpn_pointwise_interface.jl","rpn_prefix_grammar.jl",
        "rpn_prefix_grammar_gpu.jl","token_prefix_trie.jl","typed_pointwise_kernel_lowering.jl",
        "test_rpn_prefix_grammar_gpu.jl","test_rpn_native_tokens_gpu.jl","test_rpn_disabled_sampler_identity.jl")
    sources=merge(Dict(String(k)=>String(v) for (k,v) in pairs(base.source_sha256)),
        Dict(f=>laptop_rpn_digest(joinpath(@__DIR__,f)) for f in extras))
    packet=(;scope="Two independent bounded Laptop infrastructure gates, NOT task-quality evidence. Kernel parity first; fresh-worker real4B disabled-hook output identity second. No generated program executed.",
        phases=["kernels","identity"],cpu_proof_commitments=proofs,
        cpu_proofs_not_deployed="Private semantic references/mutants remain ROOT-only; hashes committed here, never inference inputs",
        source_sha256=sources,portable_manifest_sha256=base.portable_manifest_sha256,
        action_token_ids=grammar.sequences,grammar=(;max_depth=8,operation_budget=32,vocabulary=248320,eos=Int(tokenizer.bpe.eos)),
        identity_messages=["system"=>"Return only the comma-separated list, with no prose.","user"=>"List the integers from1 through40."],
        decoding=(;max_tokens=32,seed=20261024,topk=20,temperature=0.6f0,top_p=0.95f0,presence=0f0),
        project_sha256=base.project_sha256,manifest_sha256=base.manifest_sha256)
    mkpath(directory);open(io->JSON3.pretty(io,packet),joinpath(directory,"conditions.json"),"w");packet
end

function laptop_rpn_gate_state(directory)
    path=joinpath(directory,"conditions.json");p=JSON3.read(read(path,String))
    for (file,hash) in pairs(p.source_sha256)
        basename(String(file))==String(file)&&laptop_rpn_digest(joinpath(@__DIR__,String(file)))==hash||error("gate source drift: $file")
    end
    laptop_rpn_digest("gen/qwen35-4b-portable/portable.json")==p.portable_manifest_sha256||error("portable model drift")
    tokenizer=SmallLLMChecks.QwenText.Tokenizer("gen/qwen35-4b-portable")
    grammar=RPNPrefixGrammar.Grammar(s->SmallLLMChecks.QwenText.encode(tokenizer,s),Int(p.grammar.vocabulary),Int(p.grammar.eos);
        max_depth=Int(p.grammar.max_depth),budget=Int(p.grammar.operation_budget))
    grammar.sequences==[Int32.(s) for s in p.action_token_ids]||error("tokenizer/codebook drift")
    (;path,packet=p,tokenizer,grammar)
end

function qualify_laptop_rpn_gate(phase::Symbol;directory=joinpath(@__DIR__,"laptop_rpn_gate"))
    phase in (:kernels,:identity)||throw(ArgumentError("unknown gate phase"))
    normpath(pwd())=="/tmp/small-llm-bosgame-20261025"&&Sys.total_memory()<=40*1024^3||
        error("Laptop snapshot only; no device setup attempted")
    state=laptop_rpn_gate_state(directory);p=state.packet
    output=joinpath(directory,String(phase)*".json");ispath(output)&&error("preserve prior gate attempt")
    if phase==:identity
        kernelpath=joinpath(directory,"kernels.json");isfile(kernelpath)||error("kernel gate required first")
        k=JSON3.read(read(kernelpath,String))
        k.complete&&k.normal_final_waitidle&&k.conditions_sha256==laptop_rpn_digest(state.path)&&k.pid!=getpid()||error("fresh worker after normal kernel gate required")
    end
    let
        started=time();device=Mantle.Device("radeon");model=nothing
        report=Dict{String,Any}("complete"=>false,"phase"=>String(phase),"pid"=>getpid(),
            "conditions_sha256"=>laptop_rpn_digest(state.path),"runtime_provenance"=>inference_runtime_provenance())
        save()=open(io->JSON3.pretty(io,report),output,"w")
        try
            save()
            tests=if phase==:kernels
                @testset "Radeon RPN grammar/sampler shared arithmetic" begin
                    test_rpn_prefix_grammar_gpu(device)
                    test_rpn_native_tokens_gpu(device,state.grammar)
                    test_rpn_disabled_sampler_identity(device)
                end
            else
                model=load_portable_qwen35(SmallLLMChecks.Qwen35Experiment,"gen/qwen35-4b-portable",device)
                messages=[String(first(keys(m)))=>String(first(values(m))) for m in p.identity_messages]
                b=p.decoding
                options=(;max_tokens=Int(b.max_tokens),seed=UInt32(b.seed),topk=Int(b.topk),temperature=Float32(b.temperature),
                    top_p=Float32(b.top_p),presence=Float32(b.presence),output_chunk=nothing)
                ordinary=sampled_julia_answer(model,state.tokenizer,messages;thinking=false,options...)
                disabled=sampled_rpn_program(model,state.tokenizer,messages;enabled=false,options...)
                report["ordinary"]=ordinary;report["disabled"]=disabled;save()
                @testset "Real4B disabled RPN-hook output identity, not quality" begin
                    for field in (:answer,:code,:input_tokens,:generated_tokens,:decode_steps_executed,:decode_submissions,:truncated,:input_sha256)
                        @test getproperty(ordinary,field)==getproperty(disabled,field)
                    end
                    @test ordinary.sampling.draws==disabled.sampling.draws==b.max_tokens+1
                    @test ordinary.decode_submissions==disabled.decode_submissions==b.max_tokens
                    @test disabled.grammar.operations==0
                    @test disabled.grammar.done==0
                    @test disabled.grammar.rejected==0
                    @test !disabled.grammar.enabled
                end
            end
            c=Test.get_test_counts(tests);report["passed"]=c.passes+c.cumulative_passes;save()
        finally
            try model===nothing||close(model)
            finally finish_device!(device) end
        end
        report["normal_final_waitidle"]=true;report["complete"]=true;report["elapsed_seconds"]=time()-started;save();report
    end
end
