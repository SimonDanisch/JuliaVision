@testset "Convolution staging capability API compatibility" begin
    if isdefined(Mantle,:GEMM_TILINGS)
        for mode in (:scalar,:gather), c in Mantle.GEMM_TILINGS
            expected = if isdefined(Mantle,:staged_gemm_supported)
                Mantle.staged_gemm_supported(c,mode)
            else
                kernels=mode===:scalar ? Mantle.GEMM_STAGED_KERNELS : Mantle.GEMM_STAGED_GATHER_KERNELS
                haskey(kernels,c)
            end
            @test DNNKernels.conv_staged_supported(c,mode)==expected
        end
    else
        @test_skip false # no Vulkan staged-GEMM family on this installation
    end
end
