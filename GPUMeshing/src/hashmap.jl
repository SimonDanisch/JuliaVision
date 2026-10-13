"""
    HashMap(dev, n)

cumesh's hash map for `n` grid coordinates: open addressing with linear probing
over `2n` slots, UInt32 keys (`typemax` for an empty slot) and Int32 values (0
for absent). A key goes in by compare-and-swap on its slot; its value is written
by the thread that won the slot or found its own key there, so a key inserted
twice carries the value written last.

The keys are 32-bit where cumesh's are 64: Metal has no 64-bit compare-and-swap.
That holds any grid up to 1625^3 ([`insert!`](@ref) checks), and the slots are
the ones a 64-bit key of the same value hashes to.
"""
struct HashMap
    keys::Mantle.Buffer{UInt32,1}
    vals::Mantle.Buffer{Int32,1}
    capacity::Int32
end

function HashMap(dev::Mantle.Device, n::Int)
    cap = Int32(max(2n, 16))
    return HashMap(fill!(Mantle.Buffer(dev, UInt32, cap), typemax(UInt32)),
                   fill!(Mantle.Buffer(dev, Int32, cap), 0), cap)
end

function Mantle.free!(h::HashMap)
    Mantle.free!(h.keys)
    Mantle.free!(h.vals)
    return nothing
end

"""The slot a key hashes to: 64-bit Murmur3's finaliser, modulo the capacity."""
@inline function hashslot(key::UInt32, cap::Int32)
    k = UInt64(key)
    k ⊻= k >> 33
    k *= 0xff51afd7ed558ccd
    k ⊻= k >> 33
    k *= 0xc4ceb9fe1a85ec53
    k ⊻= k >> 33
    return Int32(k % UInt64(cap)) + Int32(1)
end

"""cumesh's `linear_probing_insert`."""
@inline function hashinsert!(keys, vals, key::UInt32, val::Int32, cap::Int32)
    slot = hashslot(key, cap)
    while true
        prev = (Atomix.@atomicreplace keys[slot] typemax(UInt32) => key).old
        if prev == typemax(UInt32) || prev == key
            @inbounds vals[slot] = val
            return nothing
        end
        slot = slot == cap ? Int32(1) : slot + Int32(1)
    end
end

"""cumesh's `linear_probing_lookup`: the value of `key`, 0 when it is not in."""
@inline function hashlookup(keys, vals, key::UInt32, cap::Int32)
    slot = hashslot(key, cap)
    while true
        @inbounds prev = keys[slot]
        prev == typemax(UInt32) && return Int32(0)
        prev == key && return @inbounds vals[slot]
        slot = slot == cap ? Int32(1) : slot + Int32(1)
    end
end

"""The key of the grid point `(x, y, z)`, zero-based, in an `R`^3 grid: cumesh's
flat index `x H D + y D + z`."""
@inline gridkey(x, y, z, R) = ((x % UInt32) * (R % UInt32) + (y % UInt32)) * (R % UInt32) + (z % UInt32)

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
    # Every key below `typemax`, the empty slot's.
    widen(R)^3 <= typemax(UInt32) ||
        throw(ArgumentError("HashMap: a $(R)^3 grid needs keys past 32 bits; the largest is 1625^3"))
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
