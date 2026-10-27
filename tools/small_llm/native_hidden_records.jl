using JSON3, SHA

"""Load only bounded raw Float32 matrices and token IDs, never Julia Serialization."""
function read_native_hidden_records(directory,packet,vocabulary;registration_sha256)
    width=Int(packet.width)
    0<width<=65536 && 0<vocabulary<=typemax(Int32) || error("invalid native datastore shape")
    matrices=Matrix{Float32}[];labels=Int32[];seen=Set{String}()
    for source in packet.rows
        id=String(source.id)
        occursin(r"^[a-z_]+$",id) && !(id in seen) || error("invalid/duplicate source ID")
        push!(seen,id)
        receiptpath=joinpath(directory,id*".json")
        islink(receiptpath) && error("receipt symlink")
        filesize(receiptpath)<=1048576 || error("receipt too large")
        row=JSON3.read(read(receiptpath,String))
        row.id==id && row.registration_sha256==registration_sha256 || error("receipt identity")
        row.width==width && row.records isa Integer && !(row.records isa Bool) && 1<=row.records<=768 || error("record shape")
        row.endian_bom==Int(Base.ENDIAN_BOM) || error("endianness mismatch")
        row.labels==source.token_ids[2:end] && length(row.labels)==row.records || error("next-token alignment")
        all(x->x isa Integer && !(x isa Bool) && 0<=x<vocabulary,row.labels) || error("token ID bounds")
        row.keys_file==id*".f32" || error("unexpected raw filename")
        binary=joinpath(directory,String(row.keys_file));islink(binary) && error("raw symlink")
        filesize(binary)==4*width*Int(row.records) || error("raw byte count")
        bytes2hex(sha256(read(binary)))==row.keys_sha256 || error("raw checksum")
        data=Matrix{Float32}(undef,width,Int(row.records));read!(binary,data)
        all(isfinite,data) && all(j->sum(abs2,view(data,:,j))>0,axes(data,2)) || error("invalid hidden vectors")
        push!(matrices,data);append!(labels,Int32.(row.labels))
    end
    isempty(matrices) && error("empty native datastore")
    (;keys=hcat(matrices...),labels,width,vocabulary,registration_sha256)
end

function load_native_hidden_datastore()
    isdefined(@__MODULE__,:native_hidden_datastore_state) || include("native_hidden_datastore.jl")
    state=native_hidden_datastore_state()
    config=JSON3.read(read(joinpath(String(state.packet.model_dir),"config.json"),String))
    read_native_hidden_records(state.directory,state.packet,Int(config.text_config.vocab_size);
        registration_sha256=bytes2hex(sha256(read(state.path))))
end
