# Small synthetic weights exercise the production graph on CPU without loading
# a multi-billion-parameter checkpoint. This is integration, not model parity.
using Random

function synthetic_qwen35_model(device)
    Q=Qwen35Experiment;rng=MersenneTwister(3521)
    c=(;hidden_size=128,intermediate_size=256,num_hidden_layers=2,
        linear_num_value_heads=2,linear_num_key_heads=1,
        num_attention_heads=4,num_key_value_heads=1,vocab_size=256,
        max_position_embeddings=1024,rms_norm_eps=1f-6,
        rope_parameters=(;rope_theta=1f7),layer_types=["linear_attention","full_attention"])
    m=Q.Model(device,c,Dict{String,Any}(),Any[],:none)
    upload(x)=begin b=Mantle.Buffer(device,x);push!(m.owned,b);b;end
    dense(name,k,n)=m.weights[name]=Q.DenseWeight(upload(0.02f0*randn(rng,Float32,k,n)),n,k)
    try
        embedding=upload(0.02f0*randn(rng,Float32,128,256))
        m.weights["embed_tokens.weight"]=embedding
        m.weights["lm_head"]=Q.DenseWeight(embedding,256,128)
        m.weights["norm.weight"]=upload(ones(Float32,128))
        for il in 0:1
            p="layers.$il."
            for suffix in ("input_layernorm.weight","post_attention_layernorm.weight")
                m.weights[p*suffix]=upload(ones(Float32,128))
            end
            if il==0
                a=p*"linear_attn."
                dense(a*"in_proj_qkv.weight",128,512)
                dense(a*"in_proj_z.weight",128,256)
                dense(a*"in_proj_a.weight",128,2)
                dense(a*"in_proj_b.weight",128,2)
                dense(a*"out_proj.weight",256,128)
                m.weights[a*"conv1d.weight"]=upload(0.05f0*randn(rng,Float32,4,512))
                m.weights[a*"dt_bias"]=upload(zeros(Float32,2))
                m.weights[a*"A_log"]=upload(fill(-1f0,2))
                m.weights[a*"norm.weight"]=upload(ones(Float32,128))
            else
                a=p*"self_attn."
                dense(a*"q_proj.weight",128,2048)
                dense(a*"k_proj.weight",128,256)
                dense(a*"v_proj.weight",128,256)
                dense(a*"o_proj.weight",1024,128)
                for suffix in ("q_norm.weight","k_norm.weight")
                    m.weights[a*suffix]=upload(ones(Float32,256))
                end
            end
            dense(p*"mlp.gate_proj.weight",128,256)
            dense(p*"mlp.up_proj.weight",128,256)
            dense(p*"mlp.down_proj.weight",256,128)
        end
        m
    catch
        close(m);rethrow()
    end
end

function test_qwen35_attention_sessions(device)
    model=synthetic_qwen35_model(device)
    ids=[1,5,9,33,4,1,17,8]
    metadata=(;input_ids=ids,chat_ids=ids)
    try
        rows=verify_qwen35_attention_modes(model,metadata)
        for T in (Float32,Float16), attention in (:split,:grouped)
            test_qwen35_prefill(model,metadata;kv_type=T,attention)
        end
        rows
    finally
        close(model)
    end
end
