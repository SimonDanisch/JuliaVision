"""
`releaseweights!` frees a weight dict now; dropping it frees it at a collection.

Since 2026-09-24 a dropped `Mantle.Buffer` retires from its finalizer and
returns at the next `reclaim!` (the finalizer only enqueues; freeing the pool on
the finalizer thread was the `ConcurrencyViolationError`/SIGSEGV `Mantle.trim!`
records). Before that a dropped dict freed nothing, and `QwenImageRunner`'s
`release!` did exactly that. Measured over a 256x256
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

# Release what earlier tests dropped before taking a baseline. Inside the full
# suite their garbage otherwise retires mid-measurement (an allocation past the
# pool's soft cap runs `reclaim!` itself), and the count moves for reasons that
# are not this test's: 133 -> 133 across an allocation, then 2.
function drained()
    Mantle.waitidle(DEV)
    GC.gc(true)
    Mantle.reclaim!(Mantle.pool(DEV), DEV; wait = true)
    Mantle.trim!(Mantle.pool(DEV), DEV)
    ledger()
end

# A buffer's own pool region, and whether it is still on loan: precise where the
# pool-wide count is not, since other tests' garbage retires while this runs.
regionof(b::Mantle.Buffer) = Mantle.region(b.store)
onloan(r) = get(r.block.live, r.span.lower, nothing) == r.span.upper

# In a function: a testset's top-level frame keeps its locals reachable until
# the testset ends, and a buffer it still holds is not garbage.
function droppedregion()
    a = Mantle.Buffer(DEV, Float32, (1024, 1024))
    fill!(a, 1.0f0)
    Mantle.waitidle(DEV)
    regionof(a)
end

@testset "a dropped Buffer returns at a collection, releaseweights! at once" begin
    # The contract the rest of this file rests on. A `Mantle.Buffer` retires
    # from its finalizer and returns at the next `reclaim!`, so a dropped weight
    # dict is not a leak; `releaseweights!` is the release that does not wait
    # for the collector, which is what a model about to load next needs.
    # Buffers and not `KernelAbstractions.allocate`: that array lives outside
    # Mantle's pool on some backends (Metal.jl's own), while every weight
    # `toback` uploads is a Buffer everywhere.
    r = droppedregion()
    @test onloan(r)
    drained()
    @test !onloan(r)
end

@testset "releaseweights! frees buffers and the wrappers around them" begin
    q = Mantle.Buffer(DEV, UInt32, (256, 64))
    sc = Mantle.Buffer(DEV, Float32, (256,))
    plain = Mantle.Buffer(DEV, Float16, (4096,))
    fill!(q, UInt32(0)); fill!(sc, 1.0f0); fill!(plain, Float16(0))
    regions = regionof.((q, sc, plain))

    # The shape that broke it: a struct holding device memory, which is what
    # every quantised weight in this tree is.
    wrapped = DNNKernels.ConvRotQInt8Matrix(q, sc, 256, 64)
    weights = Dict{String,Any}("plain" => plain, "wrapped" => wrapped)

    @test all(onloan, regions)
    releaseweights!(weights)
    @test isempty(weights)

    # No collection: the release is the verb's, while `q`, `sc` and `plain`
    # are still referenced here.
    Mantle.waitidle(DEV)
    Mantle.reclaim!(Mantle.pool(DEV), DEV; wait = true)
    @test !any(onloan, regions)

    # A plain device array takes its backend's own free; the dict empties.
    arrays = Dict{String,Any}("a" => KernelAbstractions.allocate(BACKEND, Float32, 256))
    releaseweights!(arrays)
    @test isempty(arrays)
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
    before = drained()              # `W` is already on it; only `A` is new below

    A = DNNKernels.quantizeint8(BACKEND, W)
    @test A.q isa Mantle.Buffer
    @test A.scale isa Mantle.Buffer
    # One ledger entry per `Buffer`, which is one fewer than the `KA.allocate`
    # path takes for the same bytes.
    @test ledger() >= before + 2

    releaseweights!(Dict{String,Any}("w" => A))
    Mantle.waitidle(DEV)
    Mantle.reclaim!(Mantle.pool(DEV), DEV; wait = true)
    Mantle.trim!(Mantle.pool(DEV), DEV)
    GC.gc(true)
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

@testset "a tracking state returns its bank" begin
    # MatAnyone's memory bank is per clip and scales with resolution and
    # `mem_frames`. It was `KernelAbstractions.allocate`d and held in a struct
    # with no teardown, so every clip's bank stayed resident for the process —
    # the same fault as the weights above, one layer up.
    model = DNNKernels.Model(Dict{String,DNNKernels.Graph}(), Dict{String,Any}();
                             backend = BACKEND)
    # AFTER the model, so the only thing between this and the count below is
    # `initstate`. Taking it before made the assertion depend on what building
    # a `Model` happens to allocate.
    before = drained()

    st = DNNKernels.initstate(model, 512, 288)

    # Four buffers for the bank, three for the state's own sensory, last-mask
    # and visual readout. The readout joined them when `readmemory` became a
    # declared graph: its shape is fixed for the clip, so allocating it per
    # frame was allocating a per-clip thing per frame.
    @test length(st.bank.buffers) == 4
    @test length(st.owned) == 3
    # It took MORE ledger entries than buffers — a persistent allocation is not
    # one entry — so the count is not asserted. What matters is that it took
    # some and that `release!` gives all of them back.
    @test ledger() > before

    # The arrays ALIAS those buffers rather than being separate allocations.
    # Not `===`: `Mantle.storage` is `deviceview`, which builds a fresh wrapper
    # per call, so identity never holds and sameness has to be asked of the
    # memory. Writing through one and reading through the other is that question.
    fill!(st.bank.key, 7.0f0)
    @test all(==(7.0f0), Array(Mantle.storage(st.bank.buffers[1])))
    fill!(st.sensory, 3.0f0)
    @test all(==(3.0f0), Array(Mantle.storage(st.owned[1])))

    Mantle.release!(st)
    Mantle.waitidle(DEV)
    Mantle.reclaim!(Mantle.pool(DEV), DEV; wait = true)
    Mantle.trim!(Mantle.pool(DEV), DEV)
    @test ledger() <= before
end
