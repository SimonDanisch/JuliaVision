include("native_hidden_records.jl")
using Test

function test_native_hidden_records()
    @testset "Native hidden record integrity (CPU only)" begin
        mktempdir() do directory
            data=Float32[1 2;3 4;5 6];binary=joinpath(directory,"public_method.f32")
            open(io->write(io,data),binary,"w")
            packet=(;width=3,rows=[(;id="public_method",token_ids=[2,0,1])])
            row=(;id="public_method",registration_sha256="fixture",width=3,records=2,
                endian_bom=Int(Base.ENDIAN_BOM),labels=[0,1],keys_file="public_method.f32",
                keys_sha256=bytes2hex(sha256(read(binary))))
            receipt=joinpath(directory,"public_method.json")
            save(r)=open(io->JSON3.pretty(io,r),receipt,"w")
            load()=read_native_hidden_records(directory,packet,3;registration_sha256="fixture")
            save(row);loaded=load()
            @test loaded.keys==data
            @test loaded.labels==Int32[0,1]
            for bad in ((;row...,labels=[1,0]),(;row...,width=4),(;row...,records=769),
                (;row...,keys_file="../escape.f32"),(;row...,keys_sha256="wrong"),
                (;row...,registration_sha256="other"),(;row...,endian_bom=0))
                save(bad);@test_throws ErrorException load()
            end
            save(row);open(io->write(io,Float32[1]),binary,"w")
            @test_throws ErrorException load()
            for values in (Float32[NaN 2;3 4;5 6],zeros(Float32,3,2))
                open(io->write(io,values),binary,"w")
                save((;row...,keys_sha256=bytes2hex(sha256(read(binary)))))
                @test_throws ErrorException load()
            end
            open(io->write(io,data),binary,"w");save(row)
            @test_throws ErrorException read_native_hidden_records(directory,(;packet...,rows=vcat(packet.rows,packet.rows)),3;registration_sha256="fixture")
        end
    end
end
