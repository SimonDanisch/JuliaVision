"""
Dropping a weight dict does not free it; `releaseweights!` does.

Mantle frees a pool region on an explicit verb and NEVER from a finalizer —
`Mantle.trim!` carries the reason: a pool freed from the GC finalizer thread
gave Lava a `ConcurrencyViolationError` and then a SIGSEGV. The consequence is
that `empty!(weights)` drops the last Julia reference and the pool's ledger
still records every region as on loan, so `trim!` finds no empty block and the
driver gets nothing back.

`QwenImageRunner`'s `release!` did exactly that. Measured over a 256x256
generation, GTT+VRAM: `release!(encoder)` returned **446 MiB of 7.5 GB**, and
the denoiser then loaded on top of a conditioner that was supposed to be gone —
15582 MiB peak for one 256² image. With `releaseweights!` the same release
returns 7971 -> 619 MiB and the pipeline peaks at 8577, which is its largest
single component rather than the sum of all three.

## Why a structural walk

A weight is not always an array. The Qwen-Image conditioner is 144
`ConvRotQInt8Matrix`, each a `q` and a `scale`, and freeing only the values that
are themselves device arrays left 144 of 326 entries behind — most of the bytes.
`releasedevice!` walks fields instead, so a new quantised layout needs no entry
anywhere.
"""

using Test
using DNNKernels
using DNNKernels: releaseweights!, releasedevice!
using Mantle
using KernelAbstractions

const BACKEND = Mantle.LavaBackend()
const DEV = Mantle.todevice(BACKEND)

"""How many allocations the pool still has on loan."""
ledger() = sum(b -> length(b.live),
               Iterators.flatten(values(Mantle.pool(DEV).blocks)); init = 0)

@testset "a dropped device array does not free itself" begin
    # The premise the rest of this file rests on. If Mantle ever grows a
    # finalizer that returns pool memory, this is the assertion that says so and
    # `releaseweights!` becomes unnecessary rather than wrong.
    before = ledger()
    let a = KernelAbstractions.allocate(BACKEND, Float32, 1024, 1024)
        fill!(a, 1.0f0)
        @test ledger() > before
    end
    GC.gc(true)
    @test ledger() > before          # still on loan with no reference to it

    Mantle.reclaim!(Mantle.pool(DEV), DEV; wait = true)
    Mantle.trim!(Mantle.pool(DEV), DEV)
    @test ledger() > before          # and reclaim/trim cannot help either
end

@testset "releaseweights! frees arrays and the wrappers around them" begin
    GC.gc(true)
    before = ledger()

    q = KernelAbstractions.allocate(BACKEND, UInt32, 256, 64)
    sc = KernelAbstractions.allocate(BACKEND, Float32, 256)
    plain = KernelAbstractions.allocate(BACKEND, Float16, 4096)
    fill!(q, UInt32(0)); fill!(sc, 1.0f0); fill!(plain, Float16(0))

    # The shape that broke it: a struct holding device arrays, which is what
    # every quantised weight in this tree is.
    wrapped = DNNKernels.ConvRotQInt8Matrix(q, sc, 256, 64)
    weights = Dict{String,Any}("plain" => plain, "wrapped" => wrapped)

    @test ledger() >= before + 3
    releaseweights!(weights)
    @test isempty(weights)

    Mantle.waitidle(DEV)
    Mantle.reclaim!(Mantle.pool(DEV), DEV; wait = true)
    Mantle.trim!(Mantle.pool(DEV), DEV)
    GC.gc(true)
    # All three regions back, which is the whole point. `<=` rather than `==`
    # because this testset does not own everything the pool is holding.
    @test ledger() <= before
end

@testset "releasedevice! leaves host data alone" begin
    # It walks fields, so it must not wander into things that are not device
    # memory and must not throw on them. A host array of scalars is skipped
    # outright; everything else is walked and found to hold nothing.
    for x in (1, 1.0, "a string", :sym, nothing, (1, 2), (a = 1, b = "x"),
              zeros(Float32, 8), [1, 2, 3], Dict(:k => 1))
        @test releasedevice!(x) === nothing
    end

    # A host array of WRAPPERS is walked, because that is where a device array
    # can hide behind a non-device container.
    host = zeros(Float32, 4)
    @test releasedevice!([host, host]) === nothing
    @test host == zeros(Float32, 4)
end
