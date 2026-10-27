using JSON3,SHA
import Mantle

"""Small native test loader ONLY. Refuses a full checkpoint to bound memory.

The eventual streaming full-model packer/loader is separate. This keeps the
graph/reference gate independent of both quantization and large allocations.
"""
function load_qwen36_moe_fixture(R,directory,device)
    path=joinpath(directory,"model.safetensors")
    filesize(path)<=256*1024^2||throw(ArgumentError("Small fixture loader only"))
    c=JSON3.read(read(joinpath(directory,"config.json"),String)).text_config
    R.validate_configuration(c)
    model=R.Model(device,c,Dict{String,Any}(),Any[],:none)
    upload(x)=begin b=Mantle.Buffer(device,x);push!(model.owned,b);b;end
    try
        tensors=R.DK.readsafetensors(path;mmap=true)
        for key in sort!(collect(keys(tensors)))
            key=="lm_head.weight"||startswith(key,"model.")||error("Unexpected native fixture key")
            name=key=="lm_head.weight" ? "lm_head" : replace(key,"model."=>"";count=1)
            raw=tensors[key];eltype(raw)===Float32||error("FP32 fixture required")
            x=copy(raw)
            if ndims(x)==2&&name!="embed_tokens.weight"
                k,m=size(x);model.weights[name]=R.Q.DenseWeight(upload(x),m,k)
            elseif ndims(x)==3&&occursin(".experts.",name)
                model.weights[name]=upload(x)
            else
                if endswith(name,"input_layernorm.weight")||endswith(name,"post_attention_layernorm.weight")||
                    name=="norm.weight"||endswith(name,"q_norm.weight")||endswith(name,"k_norm.weight")
                    x.+=1f0
                elseif endswith(name,"A_log")
                    x=.-exp.(x)
                end
                model.weights[name]=upload(x)
            end
        end
        model
    catch
        close(model);rethrow()
    end
end
