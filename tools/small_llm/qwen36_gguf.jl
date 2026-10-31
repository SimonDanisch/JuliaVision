module Qwen36GGUF
import ..Qwen36MoEExperiment
import Mantle,DNNKernels
import KernelInterface as KI
using JSON3,SHA,Mmap
const R=Qwen36MoEExperiment
const Q=R.Q
const M=R.M
const DK=DNNKernels
include("qwen36_gguf_kernels.jl")

const CHECKPOINT_SHA256="97cc5853d3c163c9cb81782ffc8ee18d909c67f8a3c6b4816b98f1103125c6c6"
const CHECKPOINT_BYTES=19889904096

function tensor_bytes(f,t)
    count=prod(t.dims)
    bytes=t.typeid==0 ? 4count : t.typeid==1 ? 2count :
        t.typeid in (12,13,14) ? begin
            t.dims[1]%256==0||error("K-quant reduction extent")
            (count÷256)*(t.typeid==12 ? 144 : t.typeid==13 ? 176 : 210)
        end : error("Unsupported GGUF tensor format $(t.typeid): $(t.name)")
    offset=Int(f.data_offset)+Int(t.offset)
    0<=offset<=filesize(f.path)-bytes||error("GGUF tensor outside file: $(t.name)")
    open(f.path) do io
        Mmap.mmap(io,Vector{UInt8},bytes,offset)
    end
end

"""Rearrange Q6_K bytes into the existing four-plane kernel format, losslessly."""
function q6_planes(bytes)
    length(bytes)%210==0||throw(DimensionMismatch("Q6 byte count"))
    blocks=length(bytes)÷210
    ql=Vector{DK.UVec4}(undef,8blocks);qh=Vector{DK.UVec4}(undef,4blocks)
    sc=Vector{DK.UVec4}(undef,blocks);d=Vector{Float16}(undef,blocks)
    arrays=(reinterpret(UInt8,ql),reinterpret(UInt8,qh),reinterpret(UInt8,sc),reinterpret(UInt8,d))
    for b in 0:blocks-1
        copyto!(arrays[1],128b+1,bytes,210b+1,128)
        copyto!(arrays[2],64b+1,bytes,210b+129,64)
        copyto!(arrays[3],16b+1,bytes,210b+193,16)
        copyto!(arrays[4],2b+1,bytes,210b+209,2)
    end
    ql,qh,sc,d
end

function combine_expert_rows(gate,up,k,rows,experts)
    each=(k÷256)*rows*144
    length(gate)==length(up)==each*experts||throw(DimensionMismatch("Expert gate/up sizes"))
    combined=Vector{UInt8}(undef,2each*experts)
    for e in 0:experts-1
        copyto!(combined,2each*e+1,gate,each*e+1,each)
        copyto!(combined,2each*e+each+1,up,each*e+1,each)
    end
    combined
end

function verify_metadata(f,c)
    md=f.metadata;p="qwen35moe."
    md["general.architecture"]=="qwen35moe"||error("GGUF architecture")
    for (key,value) in (("block_count",c.num_hidden_layers),("embedding_length",c.hidden_size),
        ("expert_count",c.num_experts),("expert_used_count",c.num_experts_per_tok),
        ("expert_feed_forward_length",c.moe_intermediate_size),
        ("expert_shared_feed_forward_length",c.shared_expert_intermediate_size),
        ("attention.head_count",c.num_attention_heads),("attention.head_count_kv",c.num_key_value_heads),
        ("attention.key_length",c.head_dim),("ssm.group_count",c.linear_num_key_heads),
        ("ssm.time_step_rank",c.linear_num_value_heads),("ssm.state_size",c.linear_key_head_dim),
        ("ssm.inner_size",c.linear_num_value_heads*c.linear_value_head_dim),("ssm.conv_kernel",4),
        ("full_attention_interval",4),("rope.dimension_count",64))
        md[p*key]==value||error("GGUF config mismatch: $key")
    end
    md[p*"rope.freq_base"]==c.rope_parameters.rope_theta||error("RoPE theta")
    f["token_embd.weight"].dims==(Int(c.hidden_size),Int(c.vocab_size))||error("Embedding shape")
    f["output.weight"].dims==(Int(c.hidden_size),Int(c.vocab_size))||error("Independent output shape")
    # Refuse an unexpected quant family before any model upload.
    all(t->t.typeid in (0,1,12,13,14),values(f.tensors))||error("Unsupported tensor types")
    nothing
end

function loadmodel(path,config_path,device;verify_hash=true)
    filesize(path)==CHECKPOINT_BYTES||error("Wrong/truncated checkpoint size")
    verify_hash&&bytes2hex(open(sha256,path))!=CHECKPOINT_SHA256&&error("GGUF publisher hash mismatch")
    f=DK.readgguf(path);c=JSON3.read(read(config_path,String)).text_config
    R.validate_configuration(c);verify_metadata(f,c)
    model=R.Model(device,c,Dict{String,Any}(),Any[],:gguf_q4ks)
    upload(x)=begin b=Mantle.Buffer(device,x);push!(model.owned,b);b;end
    used=Set{String}()
    function weight(ggname;raw=false,as_matrix=false)
        t=f[ggname];push!(used,ggname);dims=t.dims;bytes=tensor_bytes(f,t)
        if t.typeid in (0,1)
            T=t.typeid==0 ? Float32 : Float16
            x=reshape(Float32.(reinterpret(T,bytes)),dims)
            data=upload(x)
            raw&&return data
            k=dims[1];m=length(x)÷k
            return Q.DenseWeight(data,m,k)
        end
        raw&&error("Unexpected packed raw parameter: $ggname")
        k=dims[1];m=prod(dims[2:end])
        t.typeid==12&&return DK.Q4KMatrix(upload(reinterpret(DK.UVec4,bytes)),m,k)
        t.typeid==13&&return Q5KMatrix(upload(reinterpret(DK.UVec4,bytes)),m,k)
        t.typeid==14&&return DK.Q6KMatrix(map(upload,q6_planes(bytes))...,m,k)
        error("Unsupported tensor")
    end
    try
        model.weights["embed_tokens.weight"]=weight("token_embd.weight")
        model.weights["norm.weight"]=weight("output_norm.weight";raw=true)
        model.weights["lm_head"]=weight("output.weight")
        for il in 0:Int(c.num_hidden_layers)-1
            p="layers.$il.";b="blk.$il."
            model.weights[p*"input_layernorm.weight"]=weight(b*"attn_norm.weight";raw=true)
            model.weights[p*"post_attention_layernorm.weight"]=weight(b*"post_attention_norm.weight";raw=true)
            for (native,gg) in (("gate.weight","ffn_gate_inp.weight"),("shared_expert_gate.weight","ffn_gate_inp_shexp.weight"),
                ("shared_expert.gate_proj.weight","ffn_gate_shexp.weight"),("shared_expert.up_proj.weight","ffn_up_shexp.weight"),
                ("shared_expert.down_proj.weight","ffn_down_shexp.weight"))
                model.weights[p*"mlp."*native]=weight(b*gg)
            end
            gt=f[b*"ffn_gate_exps.weight"];ut=f[b*"ffn_up_exps.weight"]
            gt.typeid==ut.typeid==12||error("Expected Q4_K gate/up experts")
            k,rows,ne=gt.dims
            ut.dims==gt.dims&&(k,rows,ne)==(Int(c.hidden_size),Int(c.moe_intermediate_size),Int(c.num_experts))||error("Expert gate/up shapes")
            push!(used,gt.name,ut.name)
            combined=combine_expert_rows(tensor_bytes(f,gt),tensor_bytes(f,ut),k,rows,ne)
            packed=DK.Q4KMatrix(upload(reinterpret(DK.UVec4,combined)),2rows*ne,k)
            model.weights[p*"mlp.experts.gate_up_proj"]=M.Q4Experts(packed,2rows,ne)
            dt=f[b*"ffn_down_exps.weight"]
            dt.dims==(rows,k,ne)||error("Expert down shape")
            down=weight(dt.name)
            model.weights[p*"mlp.experts.down_proj"]=down isa Q5KMatrix ? Q5Experts(down,k,ne) : M.Q4Experts(down,k,ne)
            if c.layer_types[il+1]=="linear_attention"
                for (native,gg,raw) in (("in_proj_qkv.weight","attn_qkv.weight",false),("in_proj_z.weight","attn_gate.weight",false),
                    ("in_proj_a.weight","ssm_alpha.weight",false),("in_proj_b.weight","ssm_beta.weight",false),
                    ("out_proj.weight","ssm_out.weight",false),("A_log","ssm_a",true),("dt_bias","ssm_dt.bias",true),
                    ("conv1d.weight","ssm_conv1d.weight",true),("norm.weight","ssm_norm.weight",true))
                    model.weights[p*"linear_attn."*native]=weight(b*gg;raw)
                end
            else
                for (native,gg,raw) in (("q_proj.weight","attn_q.weight",false),("k_proj.weight","attn_k.weight",false),
                    ("v_proj.weight","attn_v.weight",false),("o_proj.weight","attn_output.weight",false),
                    ("q_norm.weight","attn_q_norm.weight",true),("k_norm.weight","attn_k_norm.weight",true))
                    model.weights[p*"self_attn."*native]=weight(b*gg;raw)
                end
            end
            println("Loaded GGUF layer ",il+1,"/",c.num_hidden_layers);flush(stdout)
            GC.gc(false)
        end
        used==Set(keys(f.tensors))||error("Unconsumed GGUF tensors; refuse partial architecture")
        model
    catch
        close(model);rethrow()
    end
end
end
