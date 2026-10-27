# Sparse operators use the same frontend as handwritten language-model graphs.
# Allocation versions, aliases, tuples and generic fusion belong to DNNKernels;
# only the sparse numerical operation schemas/lowerings belong to this runner.
const SparseDecoderGraph = DNNKernels.SemanticGraph
decoderweights!(::Mantle.Graph,w)=nothing
decoderweights!(g::SparseDecoderGraph,w)=DNNKernels.semanticweights!(g,w)
decodertransient(g::Mantle.Graph,T,dims...)=Mantle.Transient.Buffer(g,T,dims...)
decodertransient(g::SparseDecoderGraph,T,dims...)=decodertransient(g.mantle,T,dims...)
decoderdispatch!(g::Mantle.Graph,k,args,n;kw...)=Mantle.dispatch!(g,k,args,n;kw...)
decoderproduct!(v,g::Mantle.Graph,args...;kw...)=product!(v,g,args...;kw...)
decoderarg!(args...;kw...)=DNNKernels.semanticarg!(args...;kw...)
decodergraph(g::SparseDecoderGraph)=DNNKernels.semanticgraph(g)
decoderrunonce!(g::Mantle.Graph)=runonce!(g)
decoderrunonce!(g::SparseDecoderGraph)=runonce!(DNNKernels.emitsemantic!(g).mantle)

const DECODER_KERNELS = (
    ("linear_fp32",linear32!), ("cast_fp16",tofp16!),
    ("layer_norm",layernorm!), ("crop_rows",croprows!),
    ("channel_to_spatial_norm",spatialnorm!), ("norm_linear",normlinear!),
    ("zero",zerofill!), ("voxel_scatter",scattergrid!),
    ("sparse_gather",im2col3!), ("linear_epilogue",epilogue!))

function decoderdispatch!(g::SparseDecoderGraph,k,args,n;name)
    i=findfirst(p->last(p)===k,DECODER_KERNELS)
    i===nothing && error("No sparse decoder semantic schema for $k")
    DNNKernels.semanticop!(g,"trellis.$(first(DECODER_KERNELS[i]))",args;
        writes=k===scattergrid! ? (2,) : (1,),
        attrs=Dict{String,Any}("extent"=>n,"label"=>String(name)))
    nothing
end

function decoderproduct!(native,g::SparseDecoderGraph,C,w,B,M,N,K;name)
    s=planes(native,M,N,K)
    DNNKernels.semanticop!(g,"trellis.partial_matmul",(C,w,B);
        attrs=Dict{String,Any}("m"=>M,"n"=>N,"k"=>K,"planes"=>s,"label"=>String(name)))
    s
end

function DNNKernels.emitop!(ctx::DNNKernels.EmitCtx,op::DNNKernels.Op,
                           ::Val{Symbol("trellis.partial_matmul")})
    a=op.attrs;C,w,B=DNNKernels.semanticargument(ctx,op,a["arguments"])
    native=Val(Mantle.native_gemm_available(ctx.dev,Float16,Float16,Float32))
    s=product!(native,ctx.g,C,w,B,a["m"],a["n"],a["k"];name=a["label"])
    s==a["planes"] || error("Sparse decoder product layout changed during lowering")
    C
end
for (kind,kernel) in DECODER_KERNELS
    tag=Symbol("trellis.$kind")
    @eval function DNNKernels.emitop!(ctx::DNNKernels.EmitCtx,op::DNNKernels.Op,::Val{$(QuoteNode(tag))})
        args=DNNKernels.semanticargument(ctx,op,op.attrs["arguments"])
        Mantle.dispatch!(ctx.g,$kernel,args,op.attrs["extent"];name=op.attrs["label"])
        nothing
    end
end
