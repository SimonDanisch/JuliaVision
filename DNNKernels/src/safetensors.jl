"""
    readsafetensors(path) -> Dict{String,Array}

Minimal safetensors reader. The format is a u64 little-endian header length, a
JSON header mapping name to dtype/shape/byte range, then the raw data.

Tensors come back with their shape **reversed**: a PyTorch `(N, C, H, W)`
tensor becomes a Julia `(W, H, C, N)` array. PyTorch is row-major and Julia is
column-major, so reversing the extents makes the two agree element for element
with no transpose and no copy. Every kernel here indexes `(x, y, c, n)` as a
result, which is also the natural order for the graphics passes this shares a
graph with.
"""
function readsafetensors(path::AbstractString)
    open(path, "r") do io
        hlen = read(io, UInt64)
        header = JSON3.read(read(io, hlen))
        base = position(io)
        out = Dict{String,Any}()
        for (name, spec) in pairs(header)
            name === Symbol("__metadata__") && continue
            T = DTYPES[String(spec.dtype)]
            lo, hi = spec.data_offsets
            n = (hi - lo) ÷ sizeof(T)
            seek(io, base + lo)
            data = Vector{T}(undef, n)
            read!(io, data)
            dims = reverse(Tuple(Int.(spec.shape)))
            out[String(name)] = isempty(dims) ? fill(data[1]) : reshape(data, dims)
        end
        out
    end
end

const DTYPES = Dict("F64" => Float64, "F32" => Float32, "F16" => Float16,
                    "BF16" => UInt16, "I64" => Int64, "I32" => Int32,
                    "I16" => Int16, "I8" => Int8, "U8" => UInt8, "BOOL" => Bool,
                    # sdpa's philox seed/offset come back as empty u64 tensors
                    "U64" => UInt64, "U32" => UInt32, "U16" => UInt16,
                    # Complex: the rotary embeddings and Kokoro's iSTFT both have
                    # complex buffers in the graph, so a reference dump of their
                    # intermediates has complex tensors in it. Without these the
                    # reader dies with `KeyError: "C64"`, which reads like a
                    # corrupt file rather than a missing dtype.
                    "C64" => ComplexF32, "C128" => ComplexF64)

"""
    safetensorsnames(path) -> Vector{String}

Header only, without reading any tensor data.
"""
function safetensorsnames(path::AbstractString)
    open(path, "r") do io
        hlen = read(io, UInt64)
        header = JSON3.read(read(io, hlen))
        [String(k) for k in keys(header) if k !== Symbol("__metadata__")]
    end
end
