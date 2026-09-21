"""
Which devices the cooperative-matrix kernels are for.

`DeviceCaps.coopmat` is a fact about the HARDWARE, and its own documentation says
so: it is "a floor, not a promise about every operation … a kernel that wants
[something narrower] asks separately and carries the answer". Every matrix kernel
reachable from this package is a great deal narrower than the floor. They are
emitted against one instruction — `_lava_coopmat_load_f16_16x16_a`, with the shape
in the name — and Mantle's staged GEMM (`coopmat_gemm_shape`,
`coopmat_gemm_dispatch!`, `GEMM_TILINGS`) exists only in the tree that defines
them.

Four planners read that flag and then divided by `dev.tile`. On a device with
8-wide cooperative matrices — Metal's `simdgroup_matrix` is 8x8 — every one of
them ADMITTED the plan, and DepthAnything's convolution then reached
`Mantle.GEMM_TILINGS`, a name that is not merely empty there but undefined,
instead of taking the implicit-GEMM kernel beside it.

So the gate is asked once, by name, and each planner declines rather than throws.
No device is named here and none is needed: an 8-wide caps is built by hand, which
is what `DeviceCaps`' copy constructor is for.
"""

using Test, DNNKernels
import Mantle

const DKG = DNNKernels

# A 16-wide device, then the same one with 8-wide matrices and with none.
const CAPS16 = Mantle.DeviceCaps(true, 16, 32, 32, 32768, 1024, 48, 32)
const CAPS8  = Mantle.DeviceCaps(CAPS16; tile = 8)
const CAPS0  = Mantle.DeviceCaps(CAPS16; coopmat = false, tile = 0)

# Whether the staged kernels are in this build at all. Every POSITIVE assertion
# below is conditional on it, and that is the point rather than a concession: the
# predicate is a claim about the library as well as the device, so a machine
# without the tree must answer no to a 16-wide caps too.
const STAGED = isdefined(Mantle, :GEMM_TILINGS)

@testset "the gate itself" begin
    @test DKG.coopmatkernels(CAPS16) == STAGED
    # No tile of the right width, or no matrices at all: no on every build.
    @test !DKG.coopmatkernels(CAPS8)
    @test !DKG.coopmatkernels(CAPS0)
    # Wave64 is not the question — the staged GEMM runs on RDNA 3.5 — the TILE is.
    @test DKG.coopmatkernels(Mantle.DeviceCaps(CAPS16; subgroup = 64,
                                               coopmatsubgroup = 64)) == STAGED
    @test DKG.COOPMAT_TILE == 16
end

@testset "every planner declines an 8-wide device" begin
    # A `Buffer{Float16,2}` and not an eltype: `densematrix` asks about the ARRAY
    # type, and a graph resource is one of the forms it accepts. No device needed.
    B2 = Mantle.Buffer{Float16,2}
    # SAM 2's stem, for the convolution.
    out, w = (64, 64, 128, 1), (3, 3, 128, 128)

    for c in (CAPS8, CAPS0)
        p = DKG.mm_coopmat_plan(c, Float16, B2, B2, (256, 256), (256, 256))
        @test p isa DKG.Decline && p.reason === :nocoopmat

        p = DKG.conv_coopmat_plan(c, Float16, Float16, out, w)
        @test p isa DKG.Decline && p.reason === :nocoopmat

        # `flashcm_tiling` answers `nothing` rather than a `Decline` — it is a
        # tiling, not a plan — and it is reached directly, from here included.
        @test DKG.flashcm_tiling(c, 64, 1024, 1024) === nothing
    end

    # And that the gate is the only thing standing in the way: on a build that has
    # the kernels, the same shapes on a 16-wide device are taken.
    if STAGED
        @test DKG.mm_coopmat_plan(CAPS16, Float16, B2, B2, (256, 256), (256, 256)) isa
              DKG.MMCoopMatPlan
        @test DKG.conv_coopmat_plan(CAPS16, Float16, Float16, out, w) isa DKG.ConvCoopMatPlan
        @test DKG.flashcm_tiling(CAPS16, 64, 1024, 1024) !== nothing
    end
end
