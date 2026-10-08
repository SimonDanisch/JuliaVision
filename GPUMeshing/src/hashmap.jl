"""
    HashMap(dev, n)

cumesh's hash map for `n` grid coordinates: open addressing with linear probing
over `2n` slots, UInt64 keys (`typemax` for an empty slot) and Int32 values (0
for absent). A key goes in by compare-and-swap on its slot; its value is written
by the thread that won the slot or found its own key there, so a key inserted
twice carries the value written last.
"""
struct HashMap
    keys::Mantle.Buffer{UInt64,1}
    vals::Mantle.Buffer{Int32,1}
    capacity::Int32
end

function HashMap(dev::Mantle.Device, n::Int)
    cap = Int32(max(2n, 16))
    return HashMap(fill!(Mantle.Buffer(dev, UInt64, cap), typemax(UInt64)),
                   fill!(Mantle.Buffer(dev, Int32, cap), 0), cap)
end

function Mantle.free!(h::HashMap)
    Mantle.free!(h.keys)
    Mantle.free!(h.vals)
    return nothing
end

"""The slot a key hashes to: 64-bit Murmur3's finaliser, modulo the capacity."""
@inline function hashslot(k::UInt64, cap::Int32)
    k ⊻= k >> 33
    k *= 0xff51afd7ed558ccd
    k ⊻= k >> 33
    k *= 0xc4ceb9fe1a85ec53
    k ⊻= k >> 33
    return Int32(k % UInt64(cap)) + Int32(1)
end

"""cumesh's `linear_probing_insert`."""
@inline function hashinsert!(keys, vals, key::UInt64, val::Int32, cap::Int32)
    slot = hashslot(key, cap)
    while true
        prev = (Atomix.@atomicreplace keys[slot] typemax(UInt64) => key).old
        if prev == typemax(UInt64) || prev == key
            @inbounds vals[slot] = val
            return nothing
        end
        slot = slot == cap ? Int32(1) : slot + Int32(1)
    end
end

"""cumesh's `linear_probing_lookup`: the value of `key`, 0 when it is not in."""
@inline function hashlookup(keys, vals, key::UInt64, cap::Int32)
    slot = hashslot(key, cap)
    while true
        @inbounds prev = keys[slot]
        prev == typemax(UInt64) && return Int32(0)
        prev == key && return @inbounds vals[slot]
        slot = slot == cap ? Int32(1) : slot + Int32(1)
    end
end

"""The key of the grid point `(x, y, z)`, zero-based, in an `R`^3 grid: cumesh's
flat index `x H D + y D + z`."""
@inline gridkey(x, y, z, R) = ((x % UInt64) * (R % UInt64) + (y % UInt64)) * (R % UInt64) + (z % UInt64)

"""The value at grid point `(x, y, z)`, 0 when absent or outside the grid."""
@inline function lookupcoord(keys, vals, x, y, z, R, cap)
    (Int32(0) <= x < R && Int32(0) <= y < R && Int32(0) <= z < R) || return Int32(0)
    return hashlookup(keys, vals, gridkey(x, y, z, R), cap)
end

"""
    insert!(dev, map, coords, n, R) -> map

Every column of `coords` `(3, n)` (zero-based grid coordinates in an `R`^3 grid)
with its one-based column index as the value; cumesh's
`hashmap_insert_3d_idx_as_val_cuda`.
"""
function Base.insert!(dev::Mantle.Device, h::HashMap, coords::Mantle.Buffer, n::Int, R::Int)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, insertcoords!, (h.keys, h.vals, coords, n, Int32(R), h.capacity), n; name = "hashmap/insert")
    runonce!(g)
    return h
end

function insertcoords!(keys, vals, coords, n, R, cap)
    i = Int32(KI.get_global_id().x)
    i <= n || return nothing
    @inbounds hashinsert!(keys, vals, gridkey(coords[1, i], coords[2, i], coords[3, i], R), i, cap)
    return nothing
end
