using Mmap

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
function readsafetensors(path::AbstractString; mmap::Bool = filesize(path) > 2^32)
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
            dims = reverse(Tuple(Int.(spec.shape)))
            data = if mmap
                # File-backed, so the pages are page cache the kernel can drop
                # under pressure rather than anonymous memory it must kill for.
                # `read!` into a fresh Vector makes the whole checkpoint anon:
                # fine for SAM 2's 943 MB, fatal for K2 Horizon 32B's 64.78 GiB,
                # which together with the uploaded copy exceeded this worker's
                # 103.9 GiB cgroup cap and was OOM-killed with no Julia frame
                # naming the cause.
                #
                # Default on above 4 GiB only: below that the read is cheap and
                # an mmap'd array that outlives the `open` block is a subtler
                # object to hand a caller than a plain `Vector`.
                # What this costs is real and it is NOT recoverable with an
                # madvise hint. Faulting K2 Horizon 32B's 64.78 GiB in through
                # this mapping runs at ~1.5 GB/s against the 6.0 GB/s the same
                # file reads at with direct I/O: 46 s of a cold model build.
                # Both advisory routes were measured against a matched baseline
                # and neither moved it — `MADV_SEQUENTIAL` here 47.4 s,
                # `MADV_WILLNEED` one tensor ahead of `Model`'s upload loop
                # 46.5 s, nothing at all 46.2 s. (A 42.0 s reading of the
                # baseline sent me looking for a regression that was not there;
                # re-measure the baseline before believing a hint helped or
                # hurt.) Recovering it needs explicit buffered reads into a
                # reusable staging buffer, which costs this function its
                # whole-checkpoint-of-arrays contract, or a checkpoint that is
                # already quantized so there is half as much of it to read.
                Mmap.mmap(io, Vector{T}, n, base + lo)
            else
                seek(io, base + lo)
                d = Vector{T}(undef, n)
                read!(io, d)
                d
            end
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
