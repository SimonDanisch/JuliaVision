module AtomicReductionTests
using Test
import DNNKernels as DK
import Mantle as M

function check_reductions(backend)
    dev=M.todevice(backend)
    @testset "Independent atomic reductions on $(nameof(typeof(dev)))" begin
        x=Float32[(-1)^i*(i%997) for i in 1:65539]
        source=ones(Float32,4099)
        indices=Int32[mod(i,17) for i in eachindex(source)]
        graph=M.Graph(dev)
        input=M.Buffer(dev,x)
        bits=M.Buffer(dev,UInt32[0])
        src=M.Buffer(dev,source)
        idx=M.Buffer(dev,indices)
        dst=M.Buffer(dev,zeros(Float32,17))
        M.dispatch!(graph,DK.absmaxbits_kernel!,
            (bits,input,Int32(length(x)),Int32(65536)),65536)
        M.dispatch!(graph,DK.scatteradd_kernel!,
            (dst,src,idx,Int32(length(source))),length(source))
        plan=M.Plan(graph)
        try
            M.record!(plan)
            M.run!(plan)
            M.waitidle(dev)
            @test reinterpret(Float32,only(Array(M.storage(bits))))==maximum(abs,x)
            @test Array(M.storage(dst))==Float32[count(==(i),indices) for i in 0:16]
        finally
            M.free!(plan)
            foreach(M.free!,(input,bits,src,idx,dst))
        end
    end
end

foreach(check_reductions,M.eachbackend())
end
