"""
Dropping a weight dict does not free it; `releaseweights!` does.

A device array's finalizer only retires its buffer; the pool takes the region
back in a `reclaim!` on the owning thread, after the next submission
(`Mantle.trim!` carries why the finalizer thread never touches the pool: it
gave Lava a `ConcurrencyViolationError` and then a SIGSEGV). So a dropped weight
comes back only after a GC has finalized it. Until then `empty!(weights)` has
dropped the last Julia reference and the pool's ledger still records every
region as on loan, so `trim!` finds no empty block and the driver gets nothing
back.

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

const BACKEND = Mantle.defaultbackend()
const DEV = Mantle.todevice(BACKEND)

"""How many allocations the pool still has on loan."""
ledger() = sum(b -> length(b.live),
               Iterators.flatten(values(Mantle.pool(DEV).blocks)); init = 0)

"""
Whether a device array is a pool loan, which is all `ledger` counts. Where it is not, as on
Metal, the array owns its memory, and being freed shows on the array itself.
"""
const LOANED = let before = ledger()
    a = KernelAbstractions.allocate(BACKEND, Float32, 16)
    loaned = ledger() > before
    KernelAbstractions.unsafe_free!(a)
    loaned
end

"""
Every region already dropped back in the pool, so a count taken now changes only
with what the testset does. A region dropped after the last submission comes back
only once the open batch is submitted (`reclaim!` does not wait for unsubmitted
work), so without the `flush!` the next submission, inside the testset, returns
whatever earlier files dropped.
"""
function settle!()
    Mantle.flush!(DEV)
    Mantle.waitidle(DEV)
    Mantle.reclaim!(Mantle.pool(DEV), DEV; wait = true)
    Mantle.trim!(Mantle.pool(DEV), DEV)
end

"""An array allocated and filled, its last reference dropped on return."""
function dropone()
    a = KernelAbstractions.allocate(BACKEND, Float32, 1024, 1024)
    fill!(a, 1.0f0)
    return nothing
end

@testset "a dropped device array comes back only after a GC" begin
    # The premise the rest of this file rests on. The GC is held off while the
    # array is dropped, which is the state a dropped weight dict stays in until
    # a collection happens to run.
    if LOANED
        settle!()
        before = ledger()
        GC.enable(false)
        try
            dropone()
            settle!()
            @test ledger() > before      # no reference to it, and still on loan
        finally
            GC.enable(true)
        end
        GC.gc(true)
        settle!()
        @test ledger() <= before         # finalized, retired, reclaimed
    else
        @test_skip LOANED
    end
end

@testset "releaseweights! frees arrays and the wrappers around them" begin
    settle!()
    before = ledger()

    q = KernelAbstractions.allocate(BACKEND, UInt32, 256, 64)
    sc = KernelAbstractions.allocate(BACKEND, Float32, 256)
    plain = KernelAbstractions.allocate(BACKEND, Float16, 4096)
    fill!(q, UInt32(0)); fill!(sc, 1.0f0); fill!(plain, Float16(0))

    # The shape that broke it: a struct holding device arrays, which is what
    # every quantised weight in this tree is.
    wrapped = DNNKernels.ConvRotQInt8Matrix(q, sc, 256, 64)
    weights = Dict{String,Any}("plain" => plain, "wrapped" => wrapped)

    LOANED && @test ledger() >= before + 3
    releaseweights!(weights)
    @test isempty(weights)

    settle!()
    # All three regions back, which is the whole point. `<=` rather than `==`
    # because this testset does not own everything the pool is holding.
    if LOANED
        @test ledger() <= before
    else
        @test_throws Exception Array(plain)
        @test_throws Exception Array(q)
    end
end

@testset "a quantised weight is freed by the verb its regions take" begin
    # `quantizeint8` allocates through `Mantle.Buffer`, so what lands in the
    # weight dict is a pair of POOL REGIONS and not device arrays. Two things
    # have to be right for that to be freeable, and neither is the array branch
    # above:
    #
    #   * `Mantle.free!` is the verb. `KernelAbstractions.unsafe_free!` is
    #     Lava's, for memory allocated the other way, and a `Buffer` does not
    #     answer `isdevicearray` at all — so without a method of its own this
    #     weight would simply never be freed.
    #   * the generic FIELD WALK must not reach it. A `Buffer` carries its
    #     `dev` so it can free itself, and walking that reaches the
    #     `Vulkan.Instance`, whose destructor closure captures the `Instance`
    #     back. That cycle is a `StackOverflowError`, not a leak.
    W = DNNKernels.toback(BACKEND, randn(Float32, 256, 256))
    # Collected first: arrays earlier testsets dropped are still loans until a GC
    # finalizes them, and one inside `quantizeint8` lowered the count below.
    GC.gc(true)
    settle!()
    before = ledger()               # `W` is already on it; only `A` is new below

    A = DNNKernels.quantizeint8(BACKEND, W)
    @test A.q isa Mantle.Buffer
    @test A.scale isa Mantle.Buffer
    # One ledger entry per `Buffer`, which is one fewer than the `KA.allocate`
    # path takes for the same bytes.
    @test ledger() >= before + 2

    releaseweights!(Dict{String,Any}("w" => A))
    settle!()
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
