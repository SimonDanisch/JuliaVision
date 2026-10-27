isdefined(@__MODULE__,:JuliaTaskEval) || include("julia_eval.jl")
using JSON3, SHA

function qualify_tiled_sampler_software(;diagnostic_ir=false)
    # Execute trusted sampler sources only, NOT generated candidates. A separate
    # read-only-root bwrap process supplies package dependencies absent from the
    # frozen candidate sandbox, while hiding /dev/dri and forcing lavapipe.
    # This is DEVICE isolation, not the private-data boundary for model code.
    checks=raw"""
        function check_case(original,k;presence=0f0,draws=4)
            println("CASE n=",length(original)," k=",k);flush(stdout)
            logits=Mantle.Buffer(device,original);state=nothing;plan=nothing
            println("LOGITS ALLOCATED");flush(stdout)
            try
                state=GPUSampling.State(device,length(original);topk=k,temperature=0.6f0,
                    top_p=0.95f0,presence,seed=UInt32(20261024),fallback=0)
                println("STATE ALLOCATED");flush(stdout)
                graph=Mantle.Graph(device);Base.invokelatest(TiledGPUSampling.declare!,graph,logits,state)
                println("DECLARED");flush(stdout)
                plan=Mantle.record!(Mantle.Plan(graph;budget=Inf));seen=zeros(Int32,length(original))
                println("RECORDED");flush(stdout)
                for counter in UInt32(0):UInt32(draws-1)
                    expected=GPUSampling.reference(original;topk=k,temperature=0.6f0,
                        top_p=0.95f0,presence,seen,seed=UInt32(20261024),counter,fallback=0)
                    logits[:]=original
                    before=Mantle.submissions(device);Mantle.run!(plan);Mantle.waitidle(device)
                    @test Mantle.submissions(device)-before==1
                    @test only(Array(state.selected))==expected.selected
                    @test only(Array(state.status))==expected.status
                    @test only(Array(state.counter))==counter+UInt32(1)
                    seen[Int(expected.selected)+1]+=1
                    @test Array(state.seen)==seen
                    @test findall(==(0f0),Array(logits))==[Int(expected.selected)+1]
                end
                GPUSampling.activate!(state,false);logits[:]=original;Mantle.run!(plan)
                @test isequal(Array(logits),original)
                @test only(Array(state.counter))==draws
            finally
                plan===nothing || Mantle.free!(plan)
                state===nothing || close(state)
                Mantle.free!(logits)
            end
        end
        for n in (1,7,255,256,257,1025), k in unique((1,min(7,n),min(20,n)))
            values=Float32[mod(i*719,4093)/256-8 for i in 1:n]
            check_case(values,k;presence=0.5f0)
        end
        for values in (fill(-Inf32,7),Float32[0,NaN,1,2,3],Float32[0,Inf,1,2,3],
                Float32[-0.0,0.0,-1,0.0,-0.0,1,1])
            check_case(values,min(3,length(values)))
        end
        values=Float32[sin(i*0.13)+cos(i*0.017) for i in 1:248320]
        check_case(values,20;draws=2)
        Mantle.waitidle(device)
        """
    files=("gpu_sampling.jl","gpu_sampling_tiled.jl")
    result=mktempdir() do directory
        shim=diagnostic_ir ? raw"""
            @eval Mantle function argmap(ir::Method,sig::Type)
                println("REFLECTION_FALLBACK_METHOD=",ir)
                println("REFLECTION_FALLBACK_SIGNATURE=",sig)
                Base.show_backtrace(stdout,backtrace());flush(stdout)
                error("diagnostic-only: inference returned Method, not IRCode")
            end
            """ : ""
        program="using Test, Mantle\n@assert !ispath(\"/dev/dri\")\n"*
            "device=Mantle.Device(\"llvmpipe\")\n@assert occursin(\"llvmpipe\",lowercase(string(Mantle.devicename(device))))\n"*
            join(("include("*repr(joinpath(@__DIR__,file))*")" for file in files),"\n")*
            "\n"*shim*"\n@testset \"trusted tiled sampler\" begin\n"*checks*"\nend\nprintln(\"SAMPLER_PASSED\")\n"
        write(joinpath(directory,"runner.jl"),program)
        command=`timeout --signal=TERM --kill-after=5s 180s prlimit --as=12884901888 --cpu=150 --fsize=536870912 --core=0 -- bwrap --unshare-all --die-with-parent --new-session --ro-bind / / --dev /dev --proc /proc --tmpfs /tmp --tmpfs /run --ro-bind $directory /tmp/input --clearenv --setenv PATH /usr/bin:/bin --setenv HOME /tmp --setenv JULIA_DEPOT_PATH /home/sim/.julia --setenv JULIA_NUM_THREADS 2 --setenv OPENBLAS_NUM_THREADS 1 --setenv JULIA_GLFW_PLATFORM null --setenv VK_DRIVER_FILES /usr/share/vulkan/icd.d/lvp_icd.json --setenv VK_ICD_FILENAMES /usr/share/vulkan/icd.d/lvp_icd.json $(Base.julia_cmd()) --startup-file=no --history-file=no --project=/home/sim/Programmieren/VulkanDev --compiled-modules=existing -g0 /tmp/input/runner.jl`
        logfile=joinpath(directory,"run.log");started=time()
        process=open(logfile,"w") do io
            run(pipeline(ignorestatus(command),stdout=io,stderr=io))
        end
        diagnostic=open(io->String(read(io,65536)),logfile)
        (;passed=success(process)&&occursin("SAMPLER_PASSED",diagnostic),exitcode=process.exitcode,
            diagnostic,elapsed_seconds=time()-started)
    end
    report=(;complete=result.passed&&!diagnostic_ir,diagnostic_ir,
        scope="software Vulkan equivalence and one submission per replay; NOT Radeon performance or model-quality evidence. Diagnostic shim runs NEVER qualify production.",
        result,source_sha256=Dict(file=>bytes2hex(sha256(read(joinpath(@__DIR__,file)))) for file in (files...,"test_gpu_sampling_tiled.jl","julia_sandbox.sh")))
    output=diagnostic_ir ? "gpu_sampling_tiled_ir_diagnostic.json" : "gpu_sampling_tiled_software_validation_v6.json"
    open(io->JSON3.pretty(io,report),joinpath(@__DIR__,output),"w");report
end
