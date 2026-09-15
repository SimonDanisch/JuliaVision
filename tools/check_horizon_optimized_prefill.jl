# Pass the saved pre-fusion prepared graphs to avoid loading weights twice.
using HorizonRunner, DNNKernels, Test

function check_horizon_optimized_prefill(m,reference_graphs;tokens=512)
    name="horizon32b_prefill_bucket_$tokens"
    recorded=HorizonRunner.bucketmodel(m,name)
    base=m.model
    reference=DNNKernels.Model(Dict(name=>reference_graphs[name]),base.weights,
        base.backend,base.memevery,base.memframes,base.topk;record=false)
    k,v=get!(base.scratch,(:horizon_generation_cache,)) do
        HorizonRunner.newcache(m)
    end
    slots=collect(Int64,0:tokens-1)
    mask=HorizonRunner.maskbuffer(m,tokens;kv=tokens)
    HorizonRunner.causalmask!(mask,slots)
    args=(DNNKernels.toback(m.backend,reshape(mod.(slots,1000).+1,tokens,1)),k,v,
          DNNKernels.toback(m.backend,slots),DNNKernels.toback(m.backend,mask),
          DNNKernels.toback(m.backend,Int64[tokens-1]))
    @testset "full optimized $tokens-token prefill" begin
        fill!(k,0);fill!(v,0)
        expected=map(Array,DNNKernels.call(reference,name,args...;dims=(;kv=tokens)))
        fill!(k,0);fill!(v,0)
        got=map(Array,DNNKernels.call(recorded,name,args...;dims=(;kv=tokens)))
        # Bounds per output, from `tools/compare_prefill_routes.jl` on the 32B
        # model. Attention is the only difference from the reference here, and
        # its fp16 deviation compounds across 64 layers into the caches: the
        # measured cache maxima are 0.0139-0.0282 relative, worst at the
        # 128-token bucket and on BOTH routes, the cooperative flash one
        # included. The logits, the one output generation consumes, stay at
        # 0.0011-0.0043. Token-exact generation against the fixed graph is the
        # end-to-end gate; this is a drift alarm around it.
        for (i,(a,b)) in enumerate(zip(got,expected))
            err=maximum(abs,Float32.(a).-Float32.(b))
            rel=err/max(maximum(abs,Float32.(b)),eps(Float32))
            println("LARGE_PREFILL output=$i size=$(size(a)) finite=$(all(isfinite,a)) max_abs=$err relative_max=$rel")
            @test all(isfinite,a)
            @test rel < (i==1 ? .01 : .04)
        end
        @test argmax(first(got))==argmax(first(expected))
        for _ in 1:2
            fill!(k,0);fill!(v,0)
            repeated=map(Array,DNNKernels.call(recorded,name,args...;dims=(;kv=tokens)))
            @test repeated==got
        end
    end
    nothing
end
