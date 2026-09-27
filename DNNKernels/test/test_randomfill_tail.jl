"""
`randomfill_kernel!` writes the noise tensor and nothing after it.

A plain kernel's launch is whole workgroups, and the bounds check `@kernel` used
to insert has to be written by hand. This one was ported without it, so a
`randn` whose length is not a multiple of the group (1024 by default) wrote
noise past the end of its output, into whatever the plan placed next.

The kernel gets a window onto the front of a larger buffer, an ndrange the group
does not divide, and a canary in the rest.
"""

using Test
import DNNKernels, Mantle

function randomfilltail(backend)
    dev = Mantle.todevice(backend)
    n, pad = 99, 64
    p = Mantle.Buffer(dev, fill(-7f0, n + pad))
    state = Mantle.Buffer(dev, UInt32[0x12345678])
    for normal in (false, true)
        g = Mantle.Graph(dev)
        Mantle.dispatch!(g, DNNKernels.randomfill_kernel!,
                         (Mantle.viewof(p, n), state, Val(normal)), n; group = 64)
        Mantle.runonce!(g)
        got = Array(Mantle.storage(p))
        @testset "$(nameof(typeof(backend))), normal = $normal" begin
            @test all(!=(-7f0), got[1:n])
            @test all(==(-7f0), got[(n + 1):end])
        end
    end
    Mantle.free!(p)
    Mantle.free!(state)
end

@testset "randomfill writes nothing past its output" begin
    bes = Mantle.eachbackend()
    isempty(bes) && @info "no usable backend; skipping the random fill on the device"
    foreach(randomfilltail, bes)
end
