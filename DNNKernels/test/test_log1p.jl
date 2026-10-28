module Log1pTests
using Test, DNNKernels
import Mantle as M

@testset "declared log1p preserves small inputs" begin
    for be in M.eachbackend(), fused in (false, true)
        dev = M.Device(be)
        host = Float32[-0.75, -0.01, -1e-8, 0, 1e-8, 0.01, 1, 100]
        x = M.Buffer(dev, host)
        y = M.Buffer(dev, Float32, size(host))
        g = SemanticGraph(dev)
        if fused
            t = standardop!(g, "log1p.default", (x,); out=(Float32, size(host)))
            standardop!(g, "neg.default", (t,); out=y)
        else
            standardop!(g, "log1p.default", (x,); out=y)
        end
        plan = M.record!(M.Plan(emitsemantic!(g).mantle))
        try
            M.run!(plan)
            M.waitfor!(plan)
            want = (fused ? -1 : 1) .* log1p.(host)
            # An absolute tolerance would hide log(1 + x) rounding tiny
            # nonzero inputs to zero, the failure log1p must avoid.
            @test all(isapprox.(Array(M.storage(y)), want; rtol=3f-6, atol=0f0))
        finally
            M.free!(plan)
            M.free!(y)
            M.free!(x)
        end
    end
end
end
