using JSON3, SHA
function run_pocl_startup_probes()
    rows=Any[];root="/home/sim/Programmieren/VulkanDev"
    binary=joinpath(Sys.BINDIR,"julia")
    script=joinpath(@__DIR__,"pocl_clean_process.jl")
    for mode in ("none","revise","dnn")
        logfile=joinpath(@__DIR__,"pocl_startup_"*mode*".log")
        cmd=`timeout --kill-after=5s 90s $binary -C native --startup-file=no --compiled-modules=existing --threads=2 --project=$root $script $mode`
        # Hardware Vulkan is explicitly unavailable to this diagnostic even if
        # an import unexpectedly initializes Vulkan. PoCL uses its CPU runtime.
        cmd=addenv(cmd,"JULIA_KA_CPU_THREADS"=>"2","JULIA_PKG_PRECOMPILE_AUTO"=>"0",
            "VK_DRIVER_FILES"=>"/usr/share/vulkan/icd.d/lvp_icd.json",
            "VK_ICD_FILENAMES"=>"/usr/share/vulkan/icd.d/lvp_icd.json")
        started=time()
        process=open(logfile,"w") do log
            p=run(pipeline(ignorestatus(cmd),stdout=log,stderr=log);wait=false)
            try;wait(p);finally
                if process_running(p);kill(p);wait(p);end
            end
            p
        end
        diagnostic=open(io->String(read(io,65536)),logfile)
        push!(rows,(;mode,passed=success(process),exitcode=process.exitcode,elapsed_seconds=time()-started,diagnostic))
        println("Trusted clean startup ",mode," pass=",success(process));flush(stdout)
        mode=="none" && success(process) && break
    end
    report=(;scope="trusted fixed CPU kernel only, outside sandbox; hardware Vulkan ICDs disabled; not model scores",
        source_sha256=bytes2hex(sha256(read(script))),rows)
    open(io->JSON3.pretty(io,report),joinpath(@__DIR__,"pocl_startup_probes.json"),"w")
    report
end
