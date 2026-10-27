# The same DNNKernels IR and generic fusion pipeline used by exported Models.
# State-mutating kernels have explicit multi-output schemas. Packed products
# stay packed; their existing optimized kernels are semantic-op lowerings.
const BonsaiGraph = DNNKernels.SemanticGraph

_transient(g::Mantle.Graph,T,dims)=Mantle.Transient.Buffer(g,T,dims)
_transient(g::BonsaiGraph,T,dims)=_transient(g.mantle,T,dims)
_native_gemm!(g::Mantle.Graph,out,w,x;kw...)=Mantle.native_gemm_dispatch!(g.dev,g,out,w,x;kw...)
_coop_gemm!(g::Mantle.Graph,args...;kw...)=Mantle.coopmat_gemm_dispatch!(g,args...;kw...)
_recordneural(g::Mantle.Graph;kw...)=Mantle.record!(Mantle.Plan(g;kw...))
_recordneural(g::BonsaiGraph;kw...)=_recordneural(DNNKernels.emitsemantic!(g).mantle;kw...)

function _neuralgraph(s::BonsaiSession,dev)
    g=BonsaiGraph(dev;name="bonsai_neural")
    weights=Dict{String,Any}(k=>(v isa DNNKernels.PTQ1Matrix ? v.data : v) for (k,v) in s.model.weights)
    for (width,v) in s.model.signs
        weights["hadamard.signs.$width"]=v
    end
    DNNKernels.semanticweights!(g,weights)
    g
end

# Writes name all numerical results, including recurrent/conv state and KV
# caches. Unknown kernels refuse instead of losing a side effect in an opaque
# callback. Several kernels already fuse multiple elementary operations.
const BONSAI_SCHEMAS = (
    (DNNKernels.hadamard1024_kernel!,"hadamard",(1,)),
    (DNNKernels.ptq1_coopmat_kernel!,"packed_coop_product",(1,)),
    (DNNKernels.ptq1_mul_mm_kernel!,"packed_tile_product",(1,)),
    (DNNKernels.ptq1_getrows_kernel!,"packed_embedding",(1,)),
    (DNNKernels.depthwise_conv4_kernel!,"depthwise_conv_state",(1,2)),
    (DNNKernels.gated_delta_net_kernel!,"gated_delta_state",(1,2)),
    (dense2_gemv_kernel!,"dense_pair",(1,2)),
    (prepare_q_kernel!,"prepare_query",(1,2)),
    (prepare_kv_kernel!,"write_kv",(1,2,3,4)),
    (attention_partial_kernel!,"attention_partials",(1,2,3)),
    (attention_combine_kernel!,"attention_combine",(1,)),
    (swiglu_hadamard1024_batch_kernel!,"swiglu_hadamard",(1,)),
    (gdn_permute_hadamard1024_batch_kernel!,"gdn_hadamard",(1,)),
    (hadamard1024_out_batch_kernel!,"hadamard_copy",(1,2)),
    (weight_transpose_kernel!,"weight_transpose",(1,)),
    (dense_mul_mm_kernel!,"dense_tiled_product",(1,)),
    (dense_gemv_batch_kernel!,"dense_batched_product",(1,)),
    (dense_ab_split_kernel!,"dense_split",(1,2)),
    (depthwise_conv4_batch_kernel!,"batched_depthwise_state",(1,2)),
    (recurrent_split_batch_kernel!,"recurrent_split",(1,2,3)),
    (l2normalize_qk_batch_kernel!,"normalize_query_key",(1,2)),
    (gated_delta_state_batch_kernel!,"batched_delta_state",(1,2)),
    (gdn_norm_gate_batch_kernel!,"norm_gate",(1,)),
    (prepare_q_batch_kernel!,"batched_prepare_query",(1,2)),
    (prepare_kv_batch_kernel!,"batched_write_kv",(1,2,3,4)),
    (attention_prefill_kernel!,"attention_prefill",(1,)),
    (rmsnorm_batch_kernel!,"rms_norm",(1,)),
    (add_rmsnorm_batch_kernel!,"residual_rms_norm",(1,2)),
    (add_rmsnorm_hadamard_kernel!,"residual_rms_hadamard",(1,2,3)),
    (select_last_kernel!,"select_last",(1,)),
    (copy_kernel!,"copy",(1,)),
    (argmax_partial_kernel!,"argmax_parts",(1,2)),
    (argmax_final_kernel!,"argmax",(1,)))

# The same result positions for other DNNKernels frontends using these kernels.
for (kernel,_,writes) in BONSAI_SCHEMAS
    parentmodule(kernel)===BonsaiRunner || continue
    @eval DNNKernels.kernelresults(::typeof($kernel))=$writes
end
DNNKernels.kernelresults(::typeof(hadamard1024_out_batch_kernel!),args::Tuple)=
    args[end]===Val(false) ? (1,) : (1,2)

function _dispatch!(g::BonsaiGraph,kernel,args::Tuple,ndrange;
                    group::Int=256,name::AbstractString=string(nameof(kernel)))
    i=findfirst(s->s[1]===kernel,BONSAI_SCHEMAS)
    i===nothing && error("No Bonsai semantic schema for $kernel")
    _,kind,writes=BONSAI_SCHEMAS[i]
    # halfcopy is not a result when the compile-time copy flag is false.
    kernel===hadamard1024_out_batch_kernel! && args[end]===Val(false) && (writes=(1,))
    DNNKernels.semanticop!(g,"bonsai.$kind",args;writes,
        attrs=Dict{String,Any}("extent"=>ndrange,"group"=>group,"label"=>String(name)))
    nothing
end
for (kernel,kind,_) in BONSAI_SCHEMAS
    tag=Symbol("bonsai.$kind")
    @eval function DNNKernels.emitop!(ctx::DNNKernels.EmitCtx,op::DNNKernels.Op,::Val{$(QuoteNode(tag))})
        a=op.attrs;args=DNNKernels.semanticargument(ctx,op,a["arguments"])
        Mantle.dispatch!(ctx.g,$kernel,args,a["extent"];group=a["group"],name=a["label"])
        nothing
    end
end

function declare_argmax!(g::BonsaiGraph,best,x;name::AbstractString)
    pv=_transient(g,Float32,(ARGMAX_PARTS,));pidx=_transient(g,Int32,(ARGMAX_PARTS,))
    _dispatch!(g,argmax_partial_kernel!,(pv,pidx,x,Int32(length(x))),ARGMAX_PARTS*256;
        name=string(name,".parts"))
    _dispatch!(g,argmax_final_kernel!,(best,pv,pidx),256;name)
    nothing
end

function _native_gemm!(g::BonsaiGraph,out,w,x;name)
    DNNKernels.semanticop!(g,"bonsai.native_product",(out,w,x);attrs=Dict{String,Any}("label"=>String(name)))
    true
end
function DNNKernels.emitop!(ctx::DNNKernels.EmitCtx,op::DNNKernels.Op,::Val{Symbol("bonsai.native_product")})
    out,w,x=DNNKernels.semanticargument(ctx,op,op.attrs["arguments"])
    _native_gemm!(ctx.g,out,w,x;name=op.attrs["label"])===nothing && error("Native product lowering refused")
end
function _coop_gemm!(g::BonsaiGraph,out,w,x,m,n,k;name,blk_split)
    DNNKernels.semanticop!(g,"bonsai.cooperative_product",(out,w,x);
        attrs=Dict{String,Any}("m"=>m,"n"=>n,"k"=>k,"split"=>blk_split,"label"=>String(name)))
    out
end
function DNNKernels.emitop!(ctx::DNNKernels.EmitCtx,op::DNNKernels.Op,::Val{Symbol("bonsai.cooperative_product")})
    out,w,x=DNNKernels.semanticargument(ctx,op,op.attrs["arguments"]);a=op.attrs
    _coop_gemm!(ctx.g,out,w,x,a["m"],a["n"],a["k"];name=a["label"],blk_split=a["split"])
end
