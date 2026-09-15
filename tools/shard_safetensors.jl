"""
Split a `.safetensors` file into N shards that `readsafetensors` can merge back.

    julia --project=<root> tools/shard_safetensors.jl <src.safetensors> <outdir> <n>

**Why this exists.** A Julia artifact is one tree and a GitHub release asset is
capped at 2 GiB. Hunyuan3D's denoiser is 5.7 GiB of fp16 weights, so it cannot
be published as one artifact at all. Sharding the weights and binding each shard
as its own artifact is what makes it publishable — and it also means a caller
fetches the parts it needs rather than the whole model.

**Byte-level, deliberately.** This never interprets a tensor. It reads the
header, groups entries by size, and copies each tensor's raw byte range into the
shard with recomputed offsets. Going through `readsafetensors` would round-trip
every tensor through a Julia array whose dims are REVERSED (column- vs row-major),
and writing that back out means reversing again — two chances to be subtly wrong
about a non-square tensor, for no benefit. Copying bytes cannot be wrong about
shape or dtype because it never looks at either.

The format: a u64 little-endian header length, a JSON header mapping name to
dtype/shape/byte-range, then the raw data.
"""

using JSON3

function shardinfo(src::AbstractString)
    open(src, "r") do io
        hlen = read(io, UInt64)
        hdr = JSON3.read(read(io, hlen))
        base = position(io)
        entries = Tuple{String,Any}[]
        meta = nothing
        for (k, v) in pairs(hdr)
            if k === Symbol("__metadata__")
                meta = v
            else
                push!(entries, (String(k), v))
            end
        end
        # Sequential in file order, so each shard is a forward scan of the source.
        sort!(entries; by = e -> e[2].data_offsets[1])
        return (; entries, meta, base)
    end
end

"""
Group entries into `n` buckets of roughly equal bytes, preserving file order.
Greedy rather than optimal: the shards only have to fit under a cap, and keeping
file order makes every shard a sequential read of the source.
"""
function groups(entries, n::Int)
    total = sum(e[2].data_offsets[2] - e[2].data_offsets[1] for e in entries)
    target = cld(total, n)
    out, cur, cursz = Vector{typeof(entries)}(), similar(entries, 0), 0
    for e in entries
        sz = e[2].data_offsets[2] - e[2].data_offsets[1]
        if !isempty(cur) && cursz + sz > target && length(out) < n - 1
            push!(out, cur); cur = similar(entries, 0); cursz = 0
        end
        push!(cur, e); cursz += sz
    end
    isempty(cur) || push!(out, cur)
    return out
end

function writeshard(src::AbstractString, dst::AbstractString, group, base::Int)
    # Header first, with offsets renumbered from 0 for this shard.
    hdr = Dict{String,Any}()
    off = 0
    for (name, spec) in group
        sz = spec.data_offsets[2] - spec.data_offsets[1]
        hdr[name] = Dict("dtype" => String(spec.dtype),
                         "shape" => collect(Int, spec.shape),
                         "data_offsets" => [off, off + sz])
        off += sz
    end
    payload = JSON3.write(hdr)
    # safetensors pads the header to an 8-byte boundary with spaces.
    pad = (8 - (length(payload) % 8)) % 8
    payload *= " "^pad

    open(dst, "w") do out
        write(out, UInt64(length(payload)))
        write(out, payload)
        open(src, "r") do io
            buf = Vector{UInt8}(undef, 8 * 1024 * 1024)
            for (_, spec) in group
                seek(io, base + spec.data_offsets[1])
                remaining = spec.data_offsets[2] - spec.data_offsets[1]
                while remaining > 0
                    n = min(remaining, length(buf))
                    readbytes!(io, buf, n)
                    write(out, view(buf, 1:n))
                    remaining -= n
                end
            end
        end
    end
    return filesize(dst)
end

function shard(src::AbstractString, outdir::AbstractString, n::Int; stem = "weights")
    info = shardinfo(src)
    gs = groups(info.entries, n)
    mkpath(outdir)
    paths = String[]
    for (i, g) in enumerate(gs)
        dst = joinpath(outdir, "$stem-$(i)of$(length(gs)).safetensors")
        bytes = writeshard(src, dst, g, info.base)
        println("  shard $i/$(length(gs)): $(length(g)) tensors, $(round(bytes/2^30, digits=2)) GiB -> $(basename(dst))")
        push!(paths, dst)
    end
    return paths
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 3 || error("usage: shard_safetensors.jl <src> <outdir> <n>")
    shard(ARGS[1], ARGS[2], parse(Int, ARGS[3]))
end
