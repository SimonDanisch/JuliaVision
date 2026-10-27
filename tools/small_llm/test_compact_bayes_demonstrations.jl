include("compact_bayes_demonstrations.jl")
include("compact_bayes_tasks.jl")
include("compact_bayes_admission.jl")
include("portable_evidence.jl")
using Test
import Mantle

function test_compact_bayes_demonstrations(device)
    @testset "Public format examples, original problem preserved" begin
        for task in bayes_extraction_tasks()
            old=compact_bayes_messages(task);new=compact_bayes_demonstration_messages(old)
            @test first(new)==first(old)
            @test endswith(last(last(new)),"\nProblem:\n"*task.request)
            @test compact_bayes_demonstration_messages(JSON3.read(JSON3.write(old)))==new
        end
        examples=compact_bayes_demonstrations()
        for (e,target,expected) in zip(examples,("yes","blue"),(5//73,2//9))
            a=admit_compact_bayes(JSON3.write(e.network));@test a.accepted
            execution=PortableEvidence.prepare_bayes(device,a.packed.factors,a.packed.query,a.packed.observed)
            try
                before=Mantle.submissions(device);PortableEvidence.run!(execution);Mantle.waitidle(device)
                @test Mantle.submissions(device)-before==1
                @test only(Array(execution.outputs.status))==0
                index=findfirst(==(target),a.packed.query_states)
                @test isapprox(Float64(Array(execution.outputs.posterior)[index]),Float64(expected);atol=1e-6,rtol=0)
            finally close(execution) end
        end
        @test_throws ArgumentError compact_bayes_demonstration_messages(["user"=>"bad"])
        @test_throws ArgumentError compact_bayes_demonstration_messages(["system"=>"same","user"=>"no boundary"])
        @test_throws ArgumentError compact_bayes_demonstration_messages(["system"=>"same","user"=>"\nProblem:\na\nProblem:\nb"])
    end
end
