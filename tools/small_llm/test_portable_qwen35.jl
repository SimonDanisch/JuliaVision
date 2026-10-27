isdefined(@__MODULE__,:validate_portable_qwen35) || include("portable_qwen35.jl")
using Test,JSON3,SHA

function test_portable_qwen35()
    tests=@testset "Portable container rejects corrupt inputs before GPU allocation" begin
        for scenario in (:valid,:endian,:missing_file_manifest,:checksum,:offset,:bool_dimension,
                :dense_shape,:duplicate,:cache_format,:cache_coverage,:loader_source,:native_source)
            mktempdir() do root
                mkpath(joinpath(root,"cache"))
                put(path,value)=open(io->JSON3.pretty(io,value),path,"w")
                digest(path)=bytes2hex(open(sha256,path))
                for name in ("config.json","tokenizer.json","tokenizer_config.json","chat_template.jinja","LICENSE","cache/source.json")
                    write(joinpath(root,name),"PRIVATE CPU CONTAINER FIXTURE, NEVER A MODEL")
                end
                write(joinpath(root,"residual.f32"),Float32[1,2])
                write(joinpath(root,"cache/weights.bin"),zeros(UInt8,18432))
                entry=Dict("kind"=>"q4k","shape"=>[128,256],"arrays"=>[
                    Dict("offset"=>0,"dims"=>[9,128],"eltype"=>"UVec4","bytes"=>18432)])
                index=Dict("scheme"=>"qwen35_native_v1","version"=>DNNKernels.QUANTCACHE_VERSION,"entries"=>Dict("fixture"=>entry))
                put(joinpath(root,"cache/index.json"),index)
                files=Dict(file=>digest(joinpath(root,file)) for file in
                    ("config.json","tokenizer.json","tokenizer_config.json","chat_template.jinja","LICENSE","residual.f32",
                        "cache/index.json","cache/weights.bin","cache/source.json"))
                residual=Dict{String,Any}("name"=>"norm.weight","kind"=>"buffer","dims"=>[2],"offset"=>0,"bytes"=>8,
                    "m"=>0,"k"=>0,"sha256"=>digest(joinpath(root,"residual.f32")))
                cached=Dict{String,Any}("name"=>"fixture","kind"=>"cache","shape"=>[128,256])
                p=Dict{String,Any}("version"=>1,"endian_bom"=>Int(Base.ENDIAN_BOM),"records"=>[residual,cached],"files"=>files,
                    "loader_sha256"=>digest(joinpath(@__DIR__,"portable_qwen35.jl")),
                    "native_loader_sha256"=>digest(joinpath(@__DIR__,"qwen35.jl")))
                expected=""
                if scenario==:endian
                    p["endian_bom"]=0;expected="endian"
                elseif scenario==:missing_file_manifest
                    delete!(files,"cache/weights.bin");expected="incomplete portable file manifest"
                elseif scenario==:checksum
                    write(joinpath(root,"residual.f32"),Float32[3,4]);expected="checksum"
                elseif scenario==:offset
                    residual["offset"]=4;expected="residual extent"
                elseif scenario==:bool_dimension
                    residual["dims"]=Any[true,2];expected="weight dimensions"
                elseif scenario==:dense_shape
                    residual["kind"]="dense";residual["dims"]=[2,1];residual["m"]=1;residual["k"]=1;expected="dense shape"
                elseif scenario==:duplicate
                    cached["name"]="norm.weight";expected="weight coverage"
                elseif scenario==:cache_format
                    index["version"]=-1;put(joinpath(root,"cache/index.json"),index)
                    files["cache/index.json"]=digest(joinpath(root,"cache/index.json"));expected="cache format"
                elseif scenario==:cache_coverage
                    cached["name"]="different";expected="cache coverage"
                elseif scenario==:loader_source
                    p["loader_sha256"]="invalid";expected="portable loader drift"
                elseif scenario==:native_source
                    p["native_loader_sha256"]="invalid";expected="native loader drift"
                end
                put(joinpath(root,"portable.json"),p)
                result=try validate_portable_qwen35(root) catch e;e end
                if scenario==:valid
                    @test length(result.records)==2
                else
                    @test result isa ErrorException
                    @test occursin(expected,sprint(showerror,result))
                end
                @test isfile(joinpath(root,"cache/index.json"))
                @test isfile(joinpath(root,"cache/weights.bin"))
            end
        end
    end
    output=joinpath(@__DIR__,"portable_container_cpu_validation.json")
    isfile(output) && error("preserve prior qualification")
    counts=Test.get_test_counts(tests)
    report=(;complete=true,passed=counts.passes+counts.cumulative_passes,
        scope="Tiny PRIVATE CPU CONTAINER fixtures ONLY; not a valid Qwen checkpoint, GPU test or intelligence result",
        source_sha256=Dict(file=>bytes2hex(open(sha256,joinpath(@__DIR__,file))) for file in
            ("portable_qwen35.jl","test_portable_qwen35.jl")))
    open(io->JSON3.pretty(io,report),output,"w");report
end
