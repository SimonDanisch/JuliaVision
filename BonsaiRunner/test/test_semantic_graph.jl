using Test, SHA

"""Compare the DNN frontend with the preserved direct GPU declarations.

The caller supplies an already loaded checkpoint; this test never downloads one.
It checks exact logits, argmax, recurrent state and only the initialized KV
prefix. Timings exclude state readback and are diagnostics, not a speed claim.
"""
function test_bonsai_semantic_graph(BR,model;prefill_sizes=(1,3,16),decode_tokens=4)
    M=BR.Mantle;DK=BR.DNNKernels;dev=M.todevice(model.backend)
    ids=BR.encode(model.tokenizer,repeat("Julia arrays use column-major storage. Plot a sine wave. ",16))
    @assert length(ids)>=max(decode_tokens,maximum(prefill_sizes;init=1))
    records=Any[]
    onloan()=sum(block->length(block.live),Iterators.flatten(values(M.pool(dev).blocks));init=0)
    digest(x)=bytes2hex(sha256(reinterpret(UInt8,vec(Array(M.storage(x))))))
    function checkstate(a,b)
        @test a.position==b.position
        for il in sort!(collect(keys(a.recurrent)))
            @test all(digest(x)==digest(y) for (x,y) in zip(a.recurrent[il],b.recurrent[il]))
        end
        for il in sort!(collect(keys(a.kv)))
            for key in (:k,:v,:kscale,:vscale)
                x=Array(M.storage(getproperty(a.kv[il],key)))
                y=Array(M.storage(getproperty(b.kv[il],key)))
                nd=ndims(x)
                @test selectdim(x,nd,1:a.position)==selectdim(y,nd,1:b.position)
            end
        end
    end
    function pair()
        page=max(64,maximum(prefill_sizes;init=1))
        a=BR.session(model;kv_page=page)
        try
            b=BR.session(model;kv_page=page)
            try
                direct,scratch=BR._recordstep(a,dev;semantic=false)
                M.free!(a.plan);a.plan=direct;a.scratch=scratch
                return a,b
            catch
                close(b);rethrow()
            end
        catch
            close(a);rethrow()
        end
    end
    @testset "Bonsai semantic graph checkpoint parity" begin
        for nt in (0,prefill_sizes...)
            M.reclaim!(M.pool(dev),dev;wait=true)
            before=onloan()
            a,b=pair();direct=nothing
            try
                @test b.scratch.g isa DK.SemanticGraph
                @test b.scratch.g.emitted
                ir=DK.semanticgraph(b.scratch.g)
                @test count(o->o.aten=="bonsai.residual_rms_hadamard",ir.ops)==129
                @test any(o->o.aten=="dnn.ptq1_rows_pair",ir.ops)
                @test any(haskey(buf.attrs,"shapes") for buf in values(ir.buffers))
                old_times=Float64[];new_times=Float64[];counts=Tuple{Int,Int}[]
                if nt==0
                    for id in ids[1:decode_tokens]
                        M.waitidle(dev);n=M.submissions(dev)
                        push!(old_times,@elapsed begin BR.step!(a,id);M.waitidle(dev);end)
                        old_count=M.submissions(dev)-n
                        old=Array(M.storage(a.logits));oldbest=Array(M.storage(a.best))
                        n=M.submissions(dev)
                        push!(new_times,@elapsed begin BR.step!(b,id);M.waitidle(dev);end)
                        new_count=M.submissions(dev)-n
                        @test reinterpret(UInt32,old)==reinterpret(UInt32,Array(M.storage(b.logits)))
                        @test oldbest==Array(M.storage(b.best))
                        @test new_count==old_count
                        push!(counts,(old_count,new_count))
                    end
                else
                    direct=BR._recordprefill(a,nt;semantic=false)
                    direct.tokens[:]=Int32.(ids[1:nt]);direct.position[]=Int32(0)
                    M.waitidle(dev)
                    push!(old_times,@elapsed begin foreach(M.run!,direct.plans);M.waitidle(dev);end)
                    a.position=nt
                    old=Array(M.storage(a.logits));oldbest=Array(M.storage(a.best))
                    push!(new_times,@elapsed begin BR.prefill!(b,ids[1:nt]);M.waitidle(dev);end)
                    @test reinterpret(UInt32,old)==reinterpret(UInt32,Array(M.storage(b.logits)))
                    @test oldbest==Array(M.storage(b.best))
                    @test b.prefills[nt].scratch.g isa DK.SemanticGraph
                end
                checkstate(a,b)
                push!(records,(ntokens=nt,decode_tokens=nt==0 ? decode_tokens : 0,
                    old_seconds=old_times,new_seconds=new_times,submissions=counts,
                    semantic_decode_ops=length(ir.ops)))
            finally
                M.waitidle(dev)
                if direct!==nothing
                    foreach(M.free!,direct.plans);M.free!(direct.tokens);M.free!(direct.position)
                end
                close(a);close(b)
                M.reclaim!(M.pool(dev),dev;wait=true)
                @test onloan()<=before
            end
        end
    end
    records
end

function test_bonsai_semantic_failure_cleanup(BR,model)
    M=BR.Mantle;dev=M.todevice(model.backend)
    logits=M.Buffer(dev,Float32,248320);best=M.Buffer(dev,Int32,1)
    token=M.GPURef(dev,Int32(0));position=M.GPURef(dev,Int32(0))
    s=BR.BonsaiSession(model,0,Dict{Int,Tuple{Any,Any}}(),Dict{Int,Any}(),logits,best,
        nothing,token,position,nothing,64,Dict{Int,Any}())
    embedding=pop!(model.weights,"token_embd.weight",nothing)
    onloan()=sum(b->length(b.live),Iterators.flatten(values(M.pool(dev).blocks));init=0)
    @testset "Prefill declaration failure retires its inputs" begin
        try
            M.waitidle(dev);M.reclaim!(M.pool(dev),dev;wait=true);before=onloan()
            for semantic in (false,true)
                @test_throws KeyError BR._recordprefill(s,2;semantic)
                M.waitidle(dev);M.reclaim!(M.pool(dev),dev;wait=true)
                @test onloan()==before
            end
        finally
            embedding===nothing || (model.weights["token_embd.weight"]=embedding)
            close(s) # also exercises teardown before a plan was constructed
        end
    end
    nothing
end
