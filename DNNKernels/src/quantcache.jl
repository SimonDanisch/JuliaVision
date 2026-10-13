"""
Quantized weights kept on disk: written the first time a model is quantized, and
read instead of quantizing on every load after that.

A K2 Horizon 32B load reads 64.78 GiB of fp16 and quantizes it on the device; from
the cache it reads the 17.5 GiB of Q4_K that this ends up as, and none of the fp16
those weights came from: 13 s from disk and 5 s from the page cache, against 60
to 80 s quantizing (8060S).

`index.json` names, per weight key, the source's shape, the kind of quantized
matrix and the arrays it consists of: element type, dimensions and byte offset in
`weights.bin`. Every array starts on a 4096-byte boundary and is padded to one.

**Read into host memory the device imports, then copied by the device into its
own** ([`Mantle.importhost`](@ref)). One staging buffer, imported once, takes every
array in turn: a read from the file, then a device copy, which is the whole of the
upload. The arrays are not left in the imported pages for the kernels to read,
although they could be: over a working set this large the GPU's address
translation runs out on them, and a Q4_K GEMV over 18.5 GB of weights read 144
GB/s from imported huge pages against 215 GB/s from device memory (8060S). A
backend that cannot import gets an ordinary upload.

The scheme and `QUANTCACHE_VERSION` are checked and a mismatch starts the cache
over. A changed SOURCE is not detected beyond its shape: delete the directory
after re-exporting a model.
"""
const QUANTCACHE_VERSION = 1
const QUANTCACHE_ALIGN = 4096

const QUANTCACHE_ELTYPES = Dict("UVec4" => UVec4, "UInt32" => UInt32, "Float32" => Float32,
                                "Float16" => Float16)
cachename(::Type{UVec4}) = "UVec4"
cachename(::Type{UInt32}) = "UInt32"
cachename(::Type{Float32}) = "Float32"
cachename(::Type{Float16}) = "Float16"

# What a quantized matrix is made of, and how it is put back together.
cachekind(::Q4KMatrix) = "q4k"
cachekind(::QInt8Matrix) = "int8"
cachekind(::Q6KMatrix) = "q6k"
cachearrays(w::Q4KMatrix) = (w.q,)
cachearrays(w::QInt8Matrix) = (w.q, w.scale)
cachearrays(w::Q6KMatrix) = kqarrays(w)
fromcache(::Val{:q4k}, arrays, shape) = Q4KMatrix(arrays[1], shape[1], shape[2])
fromcache(::Val{:int8}, arrays, shape) = QInt8Matrix(arrays[1], arrays[2], shape[1])
fromcache(::Val{:q6k}, arrays, shape) = Q6KMatrix(arrays..., shape[1], shape[2])

mutable struct QuantCache
    dir::String
    scheme::Symbol
    entries::Dict{String,Any}
    # `weights.bin`, open for appending once an entry has been added
    writer::Union{Nothing,IOStream}
    # …and for reading once one has been loaded
    reader::Union{Nothing,IOStream}
    # The staging buffer and its import, grown to the largest array read
    staging::Union{Nothing,Tuple{Vector{UInt8},Any}}
end

"""
    QuantCache(dir, scheme) -> QuantCache

The cache in `dir` for quantization `scheme`, started over when the directory
holds another scheme or version.
"""
function QuantCache(dir::AbstractString, scheme::Symbol)
    index = joinpath(dir, "index.json")
    if isfile(index)
        j = copy(JSON3.read(read(index, String)))
        j[:version] == QUANTCACHE_VERSION && j[:scheme] == string(scheme) &&
            return QuantCache(dir, scheme, Dict{String,Any}(String(k) => v for (k, v) in j[:entries]),
                              nothing, nothing, nothing)
    end
    mkpath(dir)
    rm(index; force = true)
    rm(joinpath(dir, "weights.bin"); force = true)
    return QuantCache(dir, scheme, Dict{String,Any}(), nothing, nothing, nothing)
end

"""
    iscached(cache, key, shape, kind)

Whether the cache holds `key`, of float shape `shape`, in the format `kind`
(`:q4k`, `:q6k` or `:int8`). The format is asked so a policy that now picks
another one for a weight quantizes it again rather than load the old one.
"""
iscached(c::QuantCache, key::AbstractString, shape, kind::Symbol) =
    haskey(c.entries, key) && Tuple(Int.(c.entries[key][:shape])) == Tuple(shape) &&
    c.entries[key][:kind] == string(kind)

"""
    cachestore!(cache, key, shape, w)

Append the quantized matrix `w` of the weight `key` (whose float form is `shape`).
"""
function cachestore!(c::QuantCache, key::AbstractString, shape, w)
    if c.writer === nothing
        c.writer = open(joinpath(c.dir, "weights.bin"), "a")
        seekend(c.writer)
    end
    arrays = map(collect(cachearrays(w))) do a
        h = Array(Mantle.storage(a))
        offset = position(c.writer)
        write(c.writer, h)
        write(c.writer, zeros(UInt8, cld(sizeof(h), QUANTCACHE_ALIGN) * QUANTCACHE_ALIGN - sizeof(h)))
        Dict(:eltype => cachename(eltype(h)), :dims => collect(size(h)), :offset => offset,
             :bytes => sizeof(h))
    end
    c.entries[key] = Dict(:kind => cachekind(w), :shape => collect(shape), :arrays => arrays)
    return c
end

"""
    closecache!(cache)

Write the index if entries were added, and let go of the file and the staging
buffer.
"""
function closecache!(c::QuantCache)
    if c.writer !== nothing
        close(c.writer)
        c.writer = nothing
        open(joinpath(c.dir, "index.json"), "w") do io
            JSON3.write(io, Dict(:version => QUANTCACHE_VERSION, :scheme => string(c.scheme),
                                 :entries => c.entries))
        end
    end
    c.reader === nothing || (close(c.reader); c.reader = nothing)
    if c.staging !== nothing
        host, staged = c.staging
        c.staging = nothing
        KernelAbstractions.unsafe_free!(staged)
        finalize(host)
    end
    return c
end

# The staging buffer, imported, of at least `nbytes`.
function staging!(c::QuantCache, backend, nbytes::Int)
    if c.staging === nothing || length(c.staging[1]) < nbytes
        c.staging === nothing || (KernelAbstractions.unsafe_free!(c.staging[2]); finalize(c.staging[1]))
        host = Mantle.importablehost(UInt8, cld(nbytes, QUANTCACHE_ALIGN) * QUANTCACHE_ALIGN)
        c.staging = (host, Mantle.importhost(backend, host))
    end
    return c.staging
end

"""
    cacheload(backend, cache, key) -> quantized matrix

The weight `key` from the cache, in the device's memory.
"""
function cacheload(backend, c::QuantCache, key::AbstractString)
    e = c.entries[key]
    c.reader === nothing && (c.reader = open(joinpath(c.dir, "weights.bin")))
    arrays = map(e[:arrays]) do a
        T = QUANTCACHE_ELTYPES[a[:eltype]]
        dims = Tuple(Int.(a[:dims]))
        nbytes = Int(a[:bytes])
        seek(c.reader, Int(a[:offset]))
        if Mantle.canimporthost(backend)
            host, staged = staging!(c, backend, nbytes)
            # The previous array's copy has to be done reading before this one
            # overwrites the staging buffer.
            KernelAbstractions.synchronize(backend)
            GC.@preserve host unsafe_read(c.reader, pointer(host), nbytes)
            d = KernelAbstractions.allocate(backend, T, dims)
            copyto!(d, reshape(reinterpret(T, view(staged, 1:nbytes)), dims))
            d
        else
            h = Array{T}(undef, dims)
            GC.@preserve h unsafe_read(c.reader, Ptr{UInt8}(pointer(h)), nbytes)
            toback(backend, h)
        end
    end
    return fromcache(Val(Symbol(e[:kind])), Tuple(arrays), Tuple(Int.(e[:shape])))
end
