"""
The SwiGLU's operands are windows of ONE product.

`fuseqkv` stacks the gate and the up projection into a single matmul, so the
fused SwiGLU's two inputs are strided halves of its result. The INTERPRETED
runner has always read them in place; the declared one resolved them first,
which is two copies of a 101 MB product before a kernel that runs no faster on
the copies than on the windows — 2.43 ms against 2.57 at Qwen-Image 2.1's
`12288 x 4118`, with each copy 1.41 ms on top.
"""

using Test, DNNKernels, Mantle

const DKS = DNNKernels

# The SwiGLU's two operands are windows of ONE product, because `fuseqkv`
# stacks the gate and the up projection into a single matmul. The interpreted
# runner has always read them in place; the declared one resolved them, which
# is two copies of the product before a kernel that runs no faster on them —
# 2.43 ms against 2.57 on Qwen-Image 2.1's `12288 x 4118`, plus 1.41 ms a copy.
@testset "a stacked SwiGLU reads its halves in place" begin
    # Whatever backend is loaded, not a named one: `Mantle.LavaBackend` exists
    # only where Lava does, so naming it made this file error out rather than
    # run on a machine with a different GPU — and nothing it pins is
    # Vulkan-specific.
    backend = first(Mantle.eachbackend())
    P, N = 8, 6
    A(p...) = Dict{String,Any}(p...)
    b = Dict{String,DKS.Buffer}(
        "x"    => DKS.Buffer("x", :external, Any[N, 2P], Float32, "", (0,3), "", "", A()),
        "gate" => DKS.Buffer("gate", :view, Any[N, P], Float32, "", (0,3), "x",
                             "slice.Tensor", A("arg1" => 1, "arg2" => 0, "arg3" => P)),
        "up"   => DKS.Buffer("up", :view, Any[N, P], Float32, "", (0,3), "x",
                             "slice.Tensor", A("arg1" => 1, "arg2" => P, "arg3" => 2P)),
        "y"    => DKS.Buffer("y", :transient, Any[N, P], Float32, "", (0,3), "", "", A()))
    g = DKS.Graph("swig", String[], ["x"], ["y"], b, ["x","gate","up","y"],
                  [DKS.Op("y", "fused.swiglu", ["gate", "up"], "y", A())])
    m = DKS.Model(Dict("swig" => g), Dict{String,Any}(), backend, 1, 1, 1)
    # Julia order is reversed: the stacked product is `(2P, N)`.
    xh = Float32.(reshape(1:(2P * N), 2P, N)) ./ (4P * N)
    got = Array(only(DKS.call(m, "swig", DKS.toback(backend, xh); dims = (;))))
    want = similar(got)
    for c in 1:N, r in 1:P
        gv = xh[r, c]
        want[r, c] = Float32(Float32(gv / (1 + exp(-gv)))) * xh[P + r, c]
    end
    @test size(got) == (P, N)
    @test got ≈ want rtol=1f-6

    # And neither half is copied first: the only passes are the input's own
    # upload and the SwiGLU itself.
    plan = DKS.planfor(m.device, m.graphs["swig"], m.weights, (;); profile = true)
    DKS.replay!(plan, "swig", (DKS.toback(backend, xh),))
    names = [String(t.name) for t in Mantle.timings(plan.plan)]
    @test "y" in names
    @test count(==("y"), names) == 1
    # `materialise` names a copy after the view and its op, so these are the
    # two passes that used to be there and must not be now. (Not a substring
    # search: the input's own upload pass is called "updates".)
    @test !("gate.slice" in names)
    @test !("up.slice" in names)
end

# How many elements one thread takes, and that taking eight gives the same
# numbers as taking one.
#
# The scalar kernel moved six bytes a thread and reached 26.6 GB/s on
# Qwen-Image 2.1's `12288 x 4118` WITH THE `exp` REMOVED, so the sigmoid was
# never its ceiling and no group shape was either (12.1 ms at the best of six
# against 14.8 at the default). Eight elements a thread runs the same op at
# 104.7 GB/s — 12.06 ms to 2.90 — and sixteen gives it back, at 3.64.
@testset "a SwiGLU takes eight elements to a thread where it can" begin
    # The innermost axis has to be a run on all three operands, and its extent
    # has to divide V, because the kernel has no tail.
    @test DKS.swiglurun((16, 3), (1, 99), (1, 99), (1, 16)) == 8
    @test DKS.swiglurun((12, 3), (1, 99), (1, 99), (1, 12)) == 4
    @test DKS.swiglurun((6, 3), (1, 99), (1, 99), (1, 6)) == 2
    @test DKS.swiglurun((7, 3), (1, 99), (1, 99), (1, 7)) === nothing
    # Qwen-Image 2.1's own: two 12288-wide halves of a 24576-wide product.
    @test DKS.swiglurun((12288, 4118, 1), (1, 24576, 1), (1, 24576, 1),
                        (1, 12288, 12288 * 4118)) == 8
    # A gate that is not a run on its innermost axis is the scalar kernel's.
    @test DKS.swiglurun((16, 3), (2, 99), (1, 99), (1, 16)) === nothing
    @test DKS.swiglurun((16, 3), (1, 99), (3, 99), (1, 16)) === nothing
    @test DKS.swiglurun((), (), (), ()) === nothing

    # …and the same answer either way, on operands wide enough to take the run
    # and on operands that cannot.
    backend = first(Mantle.eachbackend())
    A(p...) = Dict{String,Any}(p...)
    for P in (8, 7)                      # 8 takes V = 8; 7 divides nothing
        N = 5
        b = Dict{String,DKS.Buffer}(
            "x"    => DKS.Buffer("x", :external, Any[N, 2P], Float32, "", (0,3), "", "", A()),
            "gate" => DKS.Buffer("gate", :view, Any[N, P], Float32, "", (0,3), "x",
                                 "slice.Tensor", A("arg1" => 1, "arg2" => 0, "arg3" => P)),
            "up"   => DKS.Buffer("up", :view, Any[N, P], Float32, "", (0,3), "x",
                                 "slice.Tensor", A("arg1" => 1, "arg2" => P, "arg3" => 2P)),
            "y"    => DKS.Buffer("y", :transient, Any[N, P], Float32, "", (0,3), "", "", A()))
        g = DKS.Graph("sw", String[], ["x"], ["y"], b, ["x","gate","up","y"],
                      [DKS.Op("y", "fused.swiglu", ["gate", "up"], "y", A())])
        m = DKS.Model(Dict("sw" => g), Dict{String,Any}(), backend, 1, 1, 1)
        xh = Float32.(reshape(1:(2P * N), 2P, N)) ./ (4P * N)
        got = Array(only(DKS.call(m, "sw", DKS.toback(backend, xh); dims = (;))))
        want = similar(got)
        for c in 1:N, r in 1:P
            gv = xh[r, c]
            want[r, c] = Float32(Float32(gv / (1 + exp(-gv)))) * xh[P + r, c]
        end
        @test size(got) == (P, N)
        @test got ≈ want rtol=1f-6
        # Still one pass either way, whichever kernel it is.
        plan = DKS.planfor(m.device, m.graphs["sw"], m.weights, (;); profile = true)
        DKS.replay!(plan, "sw", (DKS.toback(backend, xh),))
        names = [String(t.name) for t in Mantle.timings(plan.plan)]
        @test count(==("y"), names) == 1
    end
end
