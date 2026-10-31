# GPU-to-GPU migration gate; no checkpoint download or host inference engine.
using Test, Random, Statistics

function oidn_graph_testweights(OR,cin)
    rng=MersenneTwister(872);e=OR.LEVEL_CHANNELS
    weights=Dict{String,Any}()
    add(name,ci,co)=begin
        weights["$name.weight"]=Float16.(randn(rng,Float32,3,3,ci,co)./sqrt(9f0*ci))
        weights["$name.bias"]=randn(rng,Float32,co).*0.01f0
    end
    add("enc_conv1a",cin,e[1]);add("enc_conv1b",e[1],e[1])
    for i in 2:5
        add("enc_conv$(i)a",e[i-1],e[i]);add("enc_conv$(i)b",e[i],e[i])
    end
    for i in 4:-1:2
        add("dec_conv$(i)a",e[i+1]+e[i-1],e[i]);add("dec_conv$(i)b",e[i],e[i])
    end
    add("dec_conv1a",e[2]+cin,e[1]);add("dec_conv1b",e[1],e[1]);add("dec_conv1c",e[1],3)
    weights
end

function oidn_semantic_graph_checks(OR,device)
    M=OR.Mantle;be=M.backend(device);reference_device=M.todevice(be)
    rng=MersenneTwister(873);rows=Any[]
    @testset "OIDN semantic DNN graph" begin
        for cin in (3,9), (w,h) in ((32,32),(48,32))
            net=OR.UNet(oidn_graph_testweights(OR,cin),cin,be)
            plain=OR.TileBuffers(be,w,h);recorded=OR.TileBuffers(be,w,h)
            try
                for repeat in 1:3
                    M.defaultdevice!(reference_device)
                    input=Float16.(randn(rng,Float32,length(plain.input)).*0.1f0)
                    copyto!(plain.input,input);copyto!(recorded.input,input)
                    OR.forward_immediate!(be,net,plain);M.waitidle(reference_device)
                    OR.forward!(device,net,recorded)
                    M.waitidle(device)
                    valid=net.layers[:dec_conv1c].CO*w*h
                    expected=Array(plain.output)[1:valid];got=Array(recorded.output)[1:valid]
                    @test all(isfinite,got)
                    @test reinterpret(UInt16,got)==reinterpret(UInt16,expected)
                    @test length(recorded.plans)==1
                end
                p=only(values(recorded.plans))
                @test p.plan.graph.dev===device
                @test p.graph isa OR.DNNKernels.Graph
                @test length(p.graph.ops)==23
                @test count(o->o.aten=="oidn.hwc_convolution",p.graph.ops)==19
                @test count(o->o.aten=="oidn.hwc_maxpool",p.graph.ops)==4
                old=Float64[];new=Float64[];counts=Tuple{Int,Int}[]
                for trial in 1:6
                    ns=zeros(Int,2)
                    for i in (isodd(trial) ? (1,2) : (2,1))
                        owner=i==1 ? reference_device : device
                        M.defaultdevice!(owner)
                        M.waitidle(owner);a=M.submissions(owner)
                        elapsed=@elapsed begin
                            i==1 ? OR.forward_immediate!(be,net,plain) : OR.forward!(device,net,recorded)
                            M.waitidle(owner)
                        end
                        ns[i]=M.submissions(owner)-a;push!(i==1 ? old : new,elapsed)
                    end
                    push!(counts,(ns[1],ns[2]))
                end
                # A backend may already batch immediate dispatches. Require
                # one recorded submission, never assume the baseline is 23.
                @test maximum(last,counts)==1
                @test all(pair->last(pair)<=first(pair),counts)
                row=(;cin,w,h,old_ms=1e3median(old),new_ms=1e3median(new),old_samples=old,new_samples=new,submissions=counts)
                push!(rows,row);println(row);flush(stdout)
            finally
                OR.free!(recorded);OR.free!(plain);OR.free!(net)
            end
        end
    end
    rows
end

function test_oidn_semantic_graph(OR,device)
    previous=OR.Mantle.Device()
    try
        oidn_semantic_graph_checks(OR,device)
    finally
        OR.Mantle.waitidle(device)
        OR.Mantle.defaultdevice!(previous)
    end
end
