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
    GC.gc(true)
    before = ledger()               # `W` is already on it; only `A` is new below

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
    GC.gc(true)
    # AFTER the model, so the only thing between this and the count below is
    # `initstate`. Taking it before made the assertion depend on what building
    # a `Model` happens to allocate.
    before = ledger()

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
