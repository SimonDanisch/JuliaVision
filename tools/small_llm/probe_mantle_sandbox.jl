isdefined(@__MODULE__,:JuliaTaskEval) || include("julia_eval.jl")
using JSON3, SHA

function probe_mantle_sandbox()
    task=(;id="software_vulkan_isolation",category="vulkan_kernel",
        setup="println(\"PHASE=before_import\");flush(stdout);using Mantle; import KernelInterface as KI; println(\"PHASE=after_import\");flush(stdout)",tests="""
        @test !ispath("/dev/dri")
        @test !ispath("/home/sim/.ssh")
        @test !ispath("/home/sim/Programmieren/VulkanDev/dev/Mantle/.git")
        println("PHASE=before_device");flush(stdout)
        device=Mantle.Device("llvmpipe")
        @test occursin("llvmpipe",lowercase(string(Mantle.devicename(device))))
        println("SOFTWARE_DEVICE=",Mantle.devicename(device))
        x=Mantle.Buffer(device,Float32[3,-2,5,-7,9]);out=Mantle.Buffer(device,zeros(Float32,5));plan=nothing
        try
            graph=Mantle.Graph(device)
            Mantle.dispatch!(graph,double_kernel!,(out,x,Int32(5)),64;group=64)
            println("PHASE=before_record");flush(stdout)
            plan=Mantle.record!(Mantle.Plan(graph;budget=Inf))
            before=Mantle.submissions(device);Mantle.run!(plan)
            @test Mantle.submissions(device)-before==1
            @test Array(out)==Float32[6,-4,10,-14,18]
            Mantle.waitidle(device)
        finally
            plan===nothing || Mantle.free!(plan)
            Mantle.free!(x);Mantle.free!(out)
        end
        """)
    code="""
        function double_kernel!(out,x,n::Int32)
            i=Int32(KI.get_global_id().x)
            i<=n && (@inbounds out[i]=2f0*x[i])
            nothing
        end
        """
    fingerprint=bytes2hex(sha256(task.setup*task.tests*code*read(joinpath(@__DIR__,"julia_sandbox.sh"),String)))
    attempt=joinpath(@__DIR__,"mantle_sandbox_probe_"*first(fingerprint,16)*".json")
    isfile(attempt) && error("preserve previous identical probe attempt")
    result=JuliaTaskEval.evaluate(task,code;output=attempt)
    open(io->JSON3.pretty(io,result),joinpath(@__DIR__,"mantle_sandbox_probe.json"),"w")
    result.passed || error("software Vulkan sandbox not qualified; no generated kernels may run through it")
    result
end
