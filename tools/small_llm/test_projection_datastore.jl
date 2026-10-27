isdefined(@__MODULE__,:load_projection_datastore) || include("projection_datastore.jl")
using Test

function test_projection_datastore()
    @testset "Bounded raw projection datastore" begin
        @test occursin("f(x)",complete_source_prefix("f(x)=x\ng(x)=2x\nh(x)=begin"))
        @test_throws ArgumentError complete_source_prefix("f(x)=begin")
        @test_throws ArgumentError complete_source_prefix("f(x)=x";expressions=0)
        mktempdir() do directory
            model=joinpath(directory,"model");mkpath(model)
            save(path,value)=open(io->JSON3.pretty(io,value),path,"w")
            save(joinpath(model,"config.json"),(;text_config=(;vocab_size=16)))
            save(joinpath(model,"tokenizer.json"),(;test=true))
            save(joinpath(model,"model.safetensors.index.json"),(;weight_map=Dict("x"=>"weights.safetensors")))
            write(joinpath(model,"weights.safetensors"),UInt8[1,2,3])
            registration=joinpath(directory,"registration.json")
            save(registration,(;model_dir=model,checkpoint_sha256=projection_checkpoint_identity(model),
                config_sha256=bytes2hex(sha256(read(joinpath(model,"config.json")))),
                feature_ids=[1,2],rows=[(;id="test",token_ids=[2,3,4])]))
            keys=Float32[1 2;3 4];binary=joinpath(directory,"test.f32")
            open(io->write(io,keys),binary,"w")
            row=Dict{String,Any}("id"=>"test","records"=>2,"width"=>2,"keys_file"=>"test.f32",
                "keys_sha256"=>bytes2hex(sha256(read(binary))),"endian_bom"=>Int(Base.ENDIAN_BOM),
                "labels"=>[3,4],"registration_sha256"=>bytes2hex(sha256(read(registration))))
            path=joinpath(directory,"test.json");save(path,row)
            data=load_projection_datastore(;directory)
            @test data.keys==keys
            @test data.labels==Int32[3,4]
            @test data.vocabulary==16
            for (field,value) in (("records",3),("width",3),("keys_file","../test.f32"),
                    ("keys_sha256","bad"),("registration_sha256","bad"),("endian_bom",0),
                    ("labels",[4,3]),("id","other"))
                changed=copy(row);changed[field]=value;save(path,changed)
                @test_throws ErrorException load_projection_datastore(;directory)
            end
            save(path,row)
            invalid=copy(keys);invalid[1]=NaN32
            open(io->write(io,invalid),binary,"w")
            changed=copy(row);changed["keys_sha256"]=bytes2hex(sha256(read(binary)));save(path,changed)
            @test_throws ErrorException load_projection_datastore(;directory)
            write(binary,UInt8[1,2,3]);save(path,row)
            @test_throws ErrorException load_projection_datastore(;directory)
            open(io->write(io,keys),binary,"w")
            moved=joinpath(directory,"original.f32");mv(binary,moved);symlink(moved,binary)
            @test_throws ErrorException load_projection_datastore(;directory)
            write(joinpath(model,"weights.safetensors"),UInt8[4,5,6])
            @test_throws ErrorException load_projection_datastore(;directory)
        end
    end
end
