"""
    GGUFTensor

A tensor descriptor from a GGUF file. `dims` are kept in GGUF order: the first
dimension is contiguous. For a matrix weight this is `(input, output)`, while
[`PTQ1Matrix`](@ref) exposes the logical `(output, input)` shape used by a
matrix-vector product.
"""
struct GGUFTensor
    name::String
    dims::Tuple{Vararg{Int}}
    typeid::UInt32
    offset::UInt64
end

"""
    GGUFFile

The parsed metadata and tensor directory of a GGUF v2/v3 file. Tensor payloads
are mapped only when requested, so opening a multi-gigabyte checkpoint does not
read it or duplicate it in host memory.
"""
struct GGUFFile
    path::String
    version::UInt32
    metadata::Dict{String,Any}
    tensors::Dict{String,GGUFTensor}
    data_offset::UInt64
end

const GGUF_UINT8   = UInt32(0)
const GGUF_INT8    = UInt32(1)
const GGUF_UINT16  = UInt32(2)
const GGUF_INT16   = UInt32(3)
const GGUF_UINT32  = UInt32(4)
const GGUF_INT32   = UInt32(5)
const GGUF_FLOAT32 = UInt32(6)
const GGUF_BOOL    = UInt32(7)
const GGUF_STRING  = UInt32(8)
const GGUF_ARRAY   = UInt32(9)
const GGUF_UINT64  = UInt32(10)
const GGUF_INT64   = UInt32(11)
const GGUF_FLOAT64 = UInt32(12)

# ggml type ids used by Bonsai. Keeping these as raw ids is deliberate: Prism's
# PTQ formats are newer than released Julia GGUF packages and a closed enum
# makes an otherwise valid checkpoint impossible to inspect.
const GGML_TYPE_F32    = UInt32(0)
const GGML_TYPE_F16    = UInt32(1)
const GGML_TYPE_BF16   = UInt32(30)
const GGML_TYPE_PTQ1_0 = UInt32(143)

@inline function _gguf_string(io)
    n = read(io, UInt64)
    n <= UInt64(typemax(Int)) || error("GGUF string is too large")
    String(read(io, Int(n)))
end

function _gguf_value(io, typeid::UInt32)
    typeid == GGUF_UINT8   && return read(io, UInt8)
    typeid == GGUF_INT8    && return read(io, Int8)
    typeid == GGUF_UINT16  && return read(io, UInt16)
    typeid == GGUF_INT16   && return read(io, Int16)
    typeid == GGUF_UINT32  && return read(io, UInt32)
    typeid == GGUF_INT32   && return read(io, Int32)
    typeid == GGUF_FLOAT32 && return read(io, Float32)
    typeid == GGUF_BOOL    && return !iszero(read(io, UInt8))
    typeid == GGUF_STRING  && return _gguf_string(io)
    typeid == GGUF_UINT64  && return read(io, UInt64)
    typeid == GGUF_INT64   && return read(io, Int64)
    typeid == GGUF_FLOAT64 && return read(io, Float64)
    if typeid == GGUF_ARRAY
        itemtype = read(io, UInt32)
        n = read(io, UInt64)
        n <= UInt64(typemax(Int)) || error("GGUF metadata array is too large")
        return [_gguf_value(io, itemtype) for _ in 1:Int(n)]
    end
    error("unsupported GGUF metadata type $typeid")
end

"""
    readgguf(path) -> GGUFFile

Read a GGUF v2/v3 header and tensor directory without touching tensor data.
Custom tensor type ids are retained verbatim.
"""
function readgguf(path::AbstractString)
    open(path, "r") do io
        read(io, 4) == UInt8['G', 'G', 'U', 'F'] || error("$path is not a GGUF file")
        version = read(io, UInt32)
        version in (UInt32(2), UInt32(3)) || error("unsupported GGUF version $version")
        ntensors = read(io, UInt64)
        nmeta = read(io, UInt64)
        ntensors <= UInt64(typemax(Int)) || error("too many GGUF tensors")
        nmeta <= UInt64(typemax(Int)) || error("too many GGUF metadata entries")

        metadata = Dict{String,Any}()
        for _ in 1:Int(nmeta)
            key = _gguf_string(io)
            haskey(metadata, key) && error("duplicate GGUF metadata key $key")
            metadata[key] = _gguf_value(io, read(io, UInt32))
        end

        tensors = Dict{String,GGUFTensor}()
        for _ in 1:Int(ntensors)
            name = _gguf_string(io)
            ndims = read(io, UInt32)
            ndims <= UInt32(8) || error("unsupported GGUF rank $ndims for $name")
            dims = ntuple(_ -> Int(read(io, UInt64)), Int(ndims))
            typeid = read(io, UInt32)
            offset = read(io, UInt64)
            haskey(tensors, name) && error("duplicate GGUF tensor $name")
            tensors[name] = GGUFTensor(name, dims, typeid, offset)
        end

        alignment = Int(get(metadata, "general.alignment", UInt32(32)))
        alignment > 0 || error("invalid GGUF alignment $alignment")
        pos = position(io)
        data_offset = UInt64(cld(pos, alignment) * alignment)
        GGUFFile(abspath(path), version, metadata, tensors, data_offset)
    end
end

Base.getindex(f::GGUFFile, name::AbstractString) = f.tensors[String(name)]
Base.haskey(f::GGUFFile, name::AbstractString) = haskey(f.tensors, name)

function _gguf_nbytes(t::GGUFTensor)
    n = prod(t.dims)
    t.typeid == GGML_TYPE_F32  && return 4n
    t.typeid == GGML_TYPE_F16  && return 2n
    t.typeid == GGML_TYPE_BF16 && return 2n
    if t.typeid == GGML_TYPE_PTQ1_0
        n % 128 == 0 || error("PTQ1 tensor $(t.name) has $n values, not a multiple of 128")
        return (n ÷ 128) * 28
    end
    error("byte size for GGML tensor type $(t.typeid) is not implemented ($(t.name))")
end

"""Map the exact encoded byte range of a GGUF tensor without copying it."""
function ggufbytes(f::GGUFFile, tensor::Union{GGUFTensor,AbstractString})
    t = tensor isa GGUFTensor ? tensor : f[tensor]
    n = _gguf_nbytes(t)
    off = f.data_offset + t.offset
    off + UInt64(n) <= UInt64(filesize(f.path)) || error("tensor $(t.name) extends past end of GGUF")
    open(f.path, "r") do io
        Mmap.mmap(io, Vector{UInt8}, n, off)
    end
end

"""
    gguftensor(file, name)

Return a zero-copy host view for F32/F16/BF16 tensors. BF16 is represented by
its raw `UInt16` bits, matching `readsafetensors`; use `gguffloat` when a host
Float32 conversion is needed. Packed PTQ1 tensors stay byte-addressed.
"""
function gguftensor(f::GGUFFile, tensor::Union{GGUFTensor,AbstractString})
    t = tensor isa GGUFTensor ? tensor : f[tensor]
    bytes = ggufbytes(f, t)
    if t.typeid == GGML_TYPE_PTQ1_0
        return bytes
    end
    T = t.typeid == GGML_TYPE_F32 ? Float32 :
        t.typeid == GGML_TYPE_F16 ? Float16 :
        t.typeid == GGML_TYPE_BF16 ? UInt16 :
        error("unsupported dense GGML tensor type $(t.typeid)")
    reshape(reinterpret(T, bytes), t.dims)
end

"""Convert a dense GGUF tensor, including BF16, to a host Float32 array."""
function gguffloat(f::GGUFFile, tensor::Union{GGUFTensor,AbstractString})
    t = tensor isa GGUFTensor ? tensor : f[tensor]
    x = gguftensor(f, t)
    if t.typeid == GGML_TYPE_BF16
        return reshape(Float32[reinterpret(Float32, UInt32(v) << 16) for v in x], size(x))
    end
    Float32.(x)
end
