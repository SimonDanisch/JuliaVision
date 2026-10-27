isdefined(@__MODULE__,:generate_julia_answer) || include("run_julia_tasks.jl")
isdefined(@__MODULE__,:finish_device!) || include("device_finish.jl")
using Test

function verify_qwen_weight_cache(;seconds=240,model_dir="gen/qwen35-4b")
    seconds>0 || throw(ArgumentError("time budget"))
    Q=SmallLLMChecks.Qwen35Experiment;DK=Q.DK
    native=JSON3.read(read(joinpath(@__DIR__,"conversation_native_reference.json"),String))
    case=only(filter(r->r.id=="baseline_short",native.rows));ids=Int.(case.julia_ids)
    output=joinpath(@__DIR__,"qwen_quantcache_full_validation.json")
    cachebase=joinpath(model_dir,"mantle-q4k-cache")
    function digests(model)
        Dict(key=>[bytes2hex(sha256(reinterpret(UInt8,vec(Array(a))))) for a in
            (weight isa Q.DenseWeight ? (weight.data,) : weight isa DK.KQuantMatrix ? DK.cachearrays(weight) : (weight,))]
            for (key,weight) in model.weights)
    end
    let
        started=time();deadline=started+seconds;device=Mantle.Device("radeon");model=nothing;loads=Any[]
        function save(complete;checks=nothing)
            report=(;complete,scope="full-model Q4/Q6 cache correctness and setup latency ONLY, not intelligence or steady-state throughput",
                cachebase,loads,checks,elapsed_seconds=time()-started,
                source_sha256=Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in
                    ("verify_qwen_weight_cache.jl","qwen35.jl","qwen_weight_cache.jl","device_finish.jl")))
            open(io->JSON3.pretty(io,report),output,"w");report
        end
        function observe(model)
            s=Q.session(model;capacity=256,kv_type=Float32,attention=:split)
            try
                Q.prefill!(s,ids;chunk=8)
                logits=Array(s.logits)
                Q.decode!(s,8)
                tokens=Int.(Array(s.predictions)[length(ids):length(ids)+7])
                (;logits,tokens)
            finally
                close(s)
            end
        end
        try
            load_started=time()
            model=Q.loadmodel(model_dir;quantize=:q4k,device,quantcache=cachebase)
            push!(loads,(;phase="create_or_complete_source_fingerprinted_cache",seconds=time()-load_started));save(false)
            first=observe(model);hashes=digests(model)
            close(model);model=nothing;Mantle.waitidle(device)
            time()<deadline-75 || return save(false)
            load_started=time()
            model=Q.loadmodel(model_dir;quantize=:q4k,device,quantcache=cachebase)
            push!(loads,(;phase="reload_existing_cache",seconds=time()-load_started));save(false)
            second=observe(model)
            checks=(;all_weight_bytes_equal=digests(model)==hashes,
                exact_logits=isequal(first.logits,second.logits),maxabs=maximum(abs,first.logits.-second.logits),
                exact_greedy=first.tokens==second.tokens,first_tokens=first.tokens,reloaded_tokens=second.tokens)
            save(false;checks)
            @test checks.all_weight_bytes_equal
            @test checks.maxabs<2e-5
            @test checks.exact_greedy
            save(true;checks)
        finally
            try
                model===nothing || close(model)
            finally
                finish_device!(device)
            end
        end
    end
end
