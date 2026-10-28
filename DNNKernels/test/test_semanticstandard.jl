module SemanticStandardTests
using Test,Random,DNNKernels
import Mantle as M
import KernelInterface as KI
const DK=DNNKernels

function scale_kernel!(out,x,s::Float32,n::Int32)
    i=Int32(KI.get_global_id().x)
    i<=n && (@inbounds out[i]=s*x[i])
    nothing
end
DK.kernelresults(::typeof(scale_kernel!))=(1,)
function unknown_kernel!(out)
    nothing
end

bits(x)=collect(reinterpret(UInt32,vec(Array(M.storage(x)))))
names(plan)=[pp.pass.name for pp in plan.passes]
function q4(dev,rng,m,k)
    be=M.backend(dev)
    w=DK.toback(be,Float16.(0.02f0.*randn(rng,Float32,m,k)))
    try
        DK.quantizeq4k(be,w)
    finally
        DK.releasedevice!(w)
    end
end
function runplan(g)
    plan=M.record!(M.Plan(g;budget=Inf))
    M.run!(plan);M.waitfor!(plan)
    plan
end

# A handwritten MLP through standard kinds reaches the same pattern passes and
# packed lowerings as an imported graph, bit for bit with the direct kernels.
function swiglu_product(dev,rng,owned,plans)
    k,f,m=256,512,256
    Wg,Wu,Wd=q4(dev,rng,f,k),q4(dev,rng,f,k),q4(dev,rng,m,f)
    append!(owned,(Wg.q,Wu.q,Wd.q))
    x=M.Buffer(dev,randn(rng,Float32,k,1));push!(owned,x)
    out=M.Buffer(dev,Float32,(m,1));want=M.Buffer(dev,Float32,(m,1));push!(owned,out,want)
    g=SemanticGraph(dev)
    semanticweights!(g,Dict("gate"=>Wg,"up"=>Wu,"down"=>Wd))
    gate=standardop!(g,"mm.default",(x,Wg);out=(Float32,(f,1)))
    up=standardop!(g,"mm.default",(x,Wu);out=(Float32,(f,1)))
    act=standardop!(g,"fused.swiglu",(gate,up);out=(Float32,(f,1)))
    @test standardop!(g,"mm.default",(act,Wd);out)===nothing
    e=emitsemantic!(g)
    @test e.report.swiglu_products==1
    @test e.report.pinned==0
    @test count(op->op.aten=="fused.swiglumm",e.graph.ops)==1
    plan=runplan(e.mantle);push!(plans,plan)
    # Gate and up share one preparation of `x`.
    @test count(n->endswith(n,".q8act"),names(plan))==(M.caps(dev).subgroup>=64 ? 1 : 2)
    ref=M.Graph(dev);r=DK.q4krows(1)
    scratch(dims)=M.Transient.Buffer(ref,Float32,dims)
    gv,uv,av=scratch((f,1)),scratch((f,1)),scratch((f,1))
    for (o,A,input) in ((gv,Wg,x),(uv,Wu,x),(want,Wd,av))
        if A===Wd
            M.dispatch!(ref,DK.swiglu_kernel!,(av,gv,uv,Int64(f)),f)
        end
        kk=size(A,2)
        q=M.Transient.Buffer(ref,DK.UVec4,(kk÷16,1));ds=M.Transient.Buffer(ref,Float32,(kk÷16,1))
        M.dispatch!(ref,DK.q8act_kernel!,(q,ds,input,Int32(kk),Int32(kk÷32)),kk÷32)
        M.dispatch!(ref,DK.kqgemv_kernels(A)[(1,r)],
            (o,DK.kqarrays(A)...,q,ds,o,identity,Int32(size(A,1)),Int32(kk÷256),Val(false)),
            cld(cld(size(A,1),r),DK.Q4K_WG÷32)*DK.Q4K_WG;group=DK.Q4K_WG)
    end
    push!(plans,runplan(ref))
    @test bits(out)==bits(want)
end

# A product writing caller storage must drop the preparation made from it.
function bound_write(dev,rng,owned,plans)
    k=256
    W,W2=q4(dev,rng,k,k),q4(dev,rng,k,k);append!(owned,(W.q,W2.q))
    r0=randn(rng,Float32,k,1)
    R=M.Buffer(dev,r0);X=M.Buffer(dev,randn(rng,Float32,k,1))
    P=M.Buffer(dev,Float32,(k,1));Q=M.Buffer(dev,Float32,(k,1));Qref=M.Buffer(dev,Float32,(k,1))
    append!(owned,(R,X,P,Q,Qref))
    old=DK.SHARE_ACTIVATIONS[]
    try
        for (sharing,dest) in ((true,Q),(false,Qref))
            DK.SHARE_ACTIVATIONS[]=sharing
            copyto!(M.storage(R),r0)
            g=SemanticGraph(dev);semanticweights!(g,Dict("w"=>W,"w2"=>W2))
            standardop!(g,"mm.default",(R,W);out=P)
            standardop!(g,"mm.default",(X,W2);out=R)
            standardop!(g,"mm.default",(R,W);out=dest)
            push!(plans,runplan(emitsemantic!(g).mantle))
        end
    finally
        DK.SHARE_ACTIVATIONS[]=old
    end
    @test bits(Q)==bits(Qref)
    @test bits(Q)!=bits(P)
end

function contracts(dev,rng,owned,plans)
    n=16;W=q4(dev,rng,256,256);push!(owned,W.q)
    x=M.Buffer(dev,ones(Float32,n));b=M.Buffer(dev,fill(2f0,n));y=M.Buffer(dev,Float32,n)
    append!(owned,(x,b,y))
    g=SemanticGraph(dev)
    standardop!(g,"add.Tensor",(x,b);out=x)
    semantickernel!(g,scale_kernel!,(y,x,3f0,Int32(n)),n)
    e=emitsemantic!(g)
    push!(plans,runplan(e.mantle))
    @test all(==(3f0),Array(M.storage(x)))
    @test all(==(9f0),Array(M.storage(y)))
    g=SemanticGraph(dev);semanticweights!(g,Dict("w"=>W))
    a=M.Buffer(dev,Float32,(256,1));push!(owned,a)
    @test_throws ArgumentError standardop!(g,"mm.default",(a,W);out=a)
    @test_throws ArgumentError standardop!(g,"add.Tensor",(M.viewof(x,(8,);offset=8),b);out=y)
    @test_throws ArgumentError standardop!(g,"add.Tensor",(x,b);out=M.viewof(y,(8,)))
    unregistered=q4(dev,rng,256,256);push!(owned,unregistered.q)
    @test_throws ArgumentError standardop!(g,"mm.default",(a,unregistered);out=(Float32,(256,1)))
    other=standardop!(SemanticGraph(dev),"add.Tensor",(x,b);out=(Float32,(n,)))
    @test_throws ArgumentError standardop!(g,"add.Tensor",(other,b);out=y)
    @test_throws ArgumentError semantickernel!(g,scale_kernel!,(other,x,1f0,Int32(n)),n)
    @test_throws MethodError semantickernel!(g,unknown_kernel!,(y,),n)
    # A whole-tensor reshape is a standard operand; an opaque reader of the
    # whole tensor sees its own extent.
    logits=M.Buffer(dev,Float32,256);scaled=M.Buffer(dev,Float32,256);push!(owned,logits,scaled)
    g=SemanticGraph(dev);semanticweights!(g,Dict("w"=>W))
    standardop!(g,"mm.default",(a,W);out=M.viewof(logits,(256,1)))
    semantickernel!(g,scale_kernel!,(scaled,logits,2f0,Int32(256)),256)
    fill!(M.storage(a),0.5f0)
    push!(plans,runplan(emitsemantic!(g).mantle))
    @test Array(M.storage(scaled))==2f0.*Array(M.storage(logits))
    @test any(!iszero,Array(M.storage(logits)))
end

# Fusing `t = sigmoid(x)` into its consumer would read `x` after the in-place
# residual overwrote it. The hazard check pins the producer instead.
function hazards(dev,rng,owned,plans)
    n=64;xh=randn(rng,Float32,n);wh=randn(rng,Float32,n);qh=randn(rng,Float32,n)
    x,w,q,u=M.Buffer(dev,xh),M.Buffer(dev,wh),M.Buffer(dev,qh),M.Buffer(dev,Float32,n)
    y=M.Buffer(dev,Float32,n);append!(owned,(x,w,q,u,y))
    for inplace in (true,false)
        copyto!(M.storage(x),xh)
        g=SemanticGraph(dev)
        t=standardop!(g,"sigmoid.default",(x,);out=(Float32,(n,)))
        standardop!(g,"add.Tensor",(x,w);out=inplace ? x : y)
        standardop!(g,"mul.Tensor",(t,q);out=u)
        e=emitsemantic!(g)
        @test e.report.pinned==(inplace ? 1 : 0)
        @test e.report.elementwise==(inplace ? 0 : 1)
        push!(plans,runplan(e.mantle))
        @test Array(M.storage(u))≈(1f0./(1f0.+exp.(-xh))).*qh rtol=2f-6
        @test Array(M.storage(inplace ? x : y))==xh.+wh
    end
    # Folding `relu` into the product would leave the caller's `out` unwritten.
    W=q4(dev,rng,256,256);push!(owned,W.q)
    a=M.Buffer(dev,randn(rng,Float32,256,1));bias=M.Buffer(dev,randn(rng,Float32,256))
    out=M.Buffer(dev,Float32,(256,1));plain=M.Buffer(dev,Float32,(256,1));append!(owned,(a,bias,out,plain))
    g=SemanticGraph(dev);semanticweights!(g,Dict("w"=>W))
    t=standardop!(g,"addmm.default",(bias,a,W);out=(Float32,(256,1)))
    standardop!(g,"relu.default",(t,);out)
    e=emitsemantic!(g)
    @test e.report.pinned==1
    push!(plans,runplan(e.mantle))
    g=SemanticGraph(dev);semanticweights!(g,Dict("w"=>W))
    standardop!(g,"addmm.default",(bias,a,W);out=plain)
    push!(plans,runplan(emitsemantic!(g).mantle))
    @test Array(M.storage(out))==max.(Array(M.storage(plain)),0f0)
end

# `x = x + down(...)`: the residual stream after a projection. Folded, the
# product's store adds the residual; in place for one packed column, through
# scratch otherwise. Float32 sums are the same either way, so bit for bit.
function residuals(dev,rng,owned,plans)
    k,f=256,512
    W,Wg,Wu=q4(dev,rng,k,f),q4(dev,rng,f,k),q4(dev,rng,f,k);append!(owned,(W.q,Wg.q,Wu.q))
    for columns in (1,4), inplace in (true,false), swiglu in (false,true)
        xh=randn(rng,Float32,k,columns);ah=randn(rng,Float32,f,columns);nh=randn(rng,Float32,k,columns)
        results=Any[]
        for fold in (true,false)
            old=DK.FOLDRESIDUAL[];DK.FOLDRESIDUAL[]=fold
            try
                x=M.Buffer(dev,xh);a=M.Buffer(dev,ah);norm=M.Buffer(dev,nh);y=M.Buffer(dev,Float32,(k,columns))
                append!(owned,(x,a,norm,y))
                g=SemanticGraph(dev);semanticweights!(g,Dict("w"=>W,"g"=>Wg,"u"=>Wu))
                act=if swiglu
                    gv=standardop!(g,"mm.default",(norm,Wg);out=(Float32,(f,columns)))
                    uv=standardop!(g,"mm.default",(norm,Wu);out=(Float32,(f,columns)))
                    standardop!(g,"fused.swiglu",(gv,uv);out=(Float32,(f,columns)))
                else
                    a
                end
                down=standardop!(g,"mm.default",(act,W);out=(Float32,(k,columns)))
                # A 1-D residual stream seen as the product's matrix.
                r=columns==1 ? M.viewof(x,(k,1)) : x
                standardop!(g,"add.Tensor",(r,down);out=inplace ? r : y)
                e=emitsemantic!(g)
                @test e.report.residuals==(fold ? 1 : 0)
                @test e.report.pinned==0
                @test count(op->haskey(op.attrs,"residual"),e.graph.ops)==(fold ? 1 : 0)
                plan=runplan(e.mantle);push!(plans,plan)
                fold && @test !any(n->endswith(n,"residual"),names(plan)) == (columns==1 || !inplace)
                push!(results,(bits(inplace ? x : y),bits(x)))
            finally
                DK.FOLDRESIDUAL[]=old
            end
        end
        @test results[1]==results[2]
        inplace || @test results[1][2]==collect(reinterpret(UInt32,vec(xh)))
    end
end

# The same fold with a dense fp16 weight and enough columns for the cooperative
# GEMM. That kernel takes a bias as one value per ROW, broadcast across the
# columns, so the residual must not ride in it as one: handed the whole stream,
# it added the first column's residual to every column. Qwen3-Embedding's
# 28-layer decoder came out at cosine 0.31 to its reference that way, and the
# K-quant cases above never reach that kernel.
function dense_residuals(dev,rng,owned,plans)
    k,f=256,512
    wh=Float16.(0.02f0.*randn(rng,Float32,k,f))
    W=M.Buffer(dev,wh);push!(owned,W)
    for columns in (1,64), inplace in (true,false)
        xh=Float16.(randn(rng,Float32,k,columns));ah=Float16.(randn(rng,Float32,f,columns))
        results=Any[]
        for fold in (true,false)
            old=DK.FOLDRESIDUAL[];DK.FOLDRESIDUAL[]=fold
            try
                x=M.Buffer(dev,xh);a=M.Buffer(dev,ah);y=M.Buffer(dev,Float16,(k,columns))
                append!(owned,(x,a,y))
                g=SemanticGraph(dev);semanticweights!(g,Dict("w"=>W))
                down=standardop!(g,"mm.default",(a,W);out=(Float16,(k,columns)))
                r=columns==1 ? M.viewof(x,(k,1)) : x
                standardop!(g,"add.Tensor",(r,down);out=inplace ? r : y)
                e=emitsemantic!(g)
                @test e.report.residuals==(fold ? 1 : 0)
                push!(plans,runplan(e.mantle))
                push!(results,Array(M.storage(inplace ? x : y)))
            finally
                DK.FOLDRESIDUAL[]=old
            end
        end
        @test results[1]==results[2]
        # Each column gets its OWN residual.
        @test isapprox(Float32.(results[1]),Float32.(wh)*Float32.(ah).+Float32.(xh);rtol=1f-2,atol=1f-2)
    end
end

function run(dev)
    rng=MersenneTwister(5150);owned=Any[];plans=Any[]
    @testset "Standard operations in semantic graphs" begin
        try
            @testset "SwiGLU product" swiglu_product(dev,rng,owned,plans)
            @testset "Bound write invalidates Q8" bound_write(dev,rng,owned,plans)
            @testset "Contracts" contracts(dev,rng,owned,plans)
            @testset "Storage hazards" hazards(dev,rng,owned,plans)
            @testset "Residual into product" residuals(dev,rng,owned,plans)
            @testset "Residual into a dense product" dense_residuals(dev,rng,owned,plans)
        finally
            M.waitidle(dev);foreach(M.free!,plans);foreach(M.free!,owned)
        end
    end
end
end
