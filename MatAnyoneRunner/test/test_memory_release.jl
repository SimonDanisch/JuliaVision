module MemoryReleaseTests
using Test, DNNKernels, MatAnyoneRunner, Mantle
const BACKEND = Mantle.defaultbackend()
const DEV = Mantle.todevice(BACKEND)

"""How many allocations the pool still has on loan."""
ledger() = sum(b -> length(b.live),
               Iterators.flatten(values(Mantle.pool(DEV).blocks)); init = 0)

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
    settle!()
    before = ledger()

    st = MatAnyoneRunner.initstate(model, 512, 288)

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
    settle!()
    @test ledger() <= before
end

end
