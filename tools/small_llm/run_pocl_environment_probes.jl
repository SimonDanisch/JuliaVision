using JSON3
# Fixed trusted CPU program only. These controls isolate sandbox environment
# differences without relaxing the sandbox used for generated code.
function run_pocl_environment_probes()
    rows=Any[];root="/home/sim/Programmieren/VulkanDev";binary=joinpath(Sys.BINDIR,"julia")
    script=joinpath(@__DIR__,"pocl_clean_process.jl")
    for mode in ("private_depot","limited_address_space","limited_loadpath")
        mktempdir() do temporary
            logfile=joinpath(@__DIR__,"pocl_environment_"*mode*".log")
            cmd=`$binary -C native -g1 --startup-file=no --compiled-modules=existing --threads=2 --project=$root $script none`
            mode=="limited_address_space" && (cmd=`prlimit --as=12884901888 -- $cmd`)
            cmd=`timeout --kill-after=5s 90s $cmd`
            cmd=addenv(cmd,"JULIA_KA_CPU_THREADS"=>"2","JULIA_PKG_PRECOMPILE_AUTO"=>"0",
                "VK_DRIVER_FILES"=>"/usr/share/vulkan/icd.d/lvp_icd.json",
                "VK_ICD_FILENAMES"=>"/usr/share/vulkan/icd.d/lvp_icd.json")
            mode=="private_depot" && (cmd=addenv(cmd,"HOME"=>temporary,"JULIA_DEPOT_PATH"=>temporary*"/depot:/home/sim/.julia"))
            mode=="limited_loadpath" && (cmd=addenv(cmd,"JULIA_LOAD_PATH"=>"@:@stdlib"))
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
            println("Trusted environment ",mode," pass=",success(process));flush(stdout)
        end
    end
    report=(;scope="trusted CPU-only controls, outside sandbox; not generated candidates or model scores",rows)
    open(io->JSON3.pretty(io,report),joinpath(@__DIR__,"pocl_environment_probes.json"),"w")
    report
end
