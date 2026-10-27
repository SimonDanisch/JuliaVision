include("laptop_experiment_progress.jl")
using Test

function test_laptop_experiment_progress()
    @testset "Infrastructure inventory never counts as task quality" begin
        mktempdir() do root
            row()=only(laptop_infrastructure_progress(root))
            @test row().status=="not_registered"
            directory=mkpath(joinpath(root,"laptop_rpn_gate"))
            conditions=joinpath(directory,"conditions.json")
            write(conditions,"{}")
            digest=bytes2hex(sha256(read(conditions)))
            save(phase;pid=1,complete=true,idle=true,passed=phase=="kernels" ? 557 : 14,hash=digest)=
                open(io->JSON3.write(io,(;phase,pid,complete,normal_final_waitidle=idle,passed,conditions_sha256=hash)),
                    joinpath(directory,phase*".json"),"w")
            @test [r.status for r in row().phases]==["pending","pending"]
            save("identity";pid=2)
            @test row().phases[2].status=="incomplete"
            save("kernels";complete=false)
            @test row().phases[1].status=="incomplete"
            save("kernels";idle=false)
            @test row().phases[1].status=="incomplete"
            save("kernels";passed=556)
            @test row().phases[1].status=="incomplete"
            save("kernels";hash="wrong")
            @test row().phases[1].status=="invalid_receipt"
            save("kernels")
            @test row().status=="qualified"
            @test !row().task_quality_evidence
            save("identity";pid=1)
            @test row().phases[2].status=="incomplete"
            write(joinpath(directory,"identity.json"),"not JSON")
            @test row().phases[2].status=="invalid_receipt"
            write(joinpath(directory,"kernels.json"),"{}")
            @test row().phases[1].status=="invalid_receipt"
        end
    end
end
