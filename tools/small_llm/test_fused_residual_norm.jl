using Test,Random
import Mantle,BonsaiRunner

function test_fused_residual_norm(device)
    rng=MersenneTwister(3636);owned=Any[];plan=nothing
    upload(x)=begin b=Mantle.Buffer(device,x);push!(owned,b);b;end
    @testset "Residual norm covers every subgroup partial" begin
        for n in (64,256,257,2048,2560),adversarial in (false,true)
            try
                tokens=3;x=randn(rng,Float32,n,tokens);branch=randn(rng,Float32,n,tokens);weights=rand(rng,Float32,n).+0.5f0
                if adversarial
                    # For n256/subgroup8, all energy is outside partials1..8.
                    fill!(x,0f0);fill!(branch,0f0)
                    for i in 65:n;branch[i,:].=Float32(i%7+1);end
                end
                residual=x+branch;eps=1f-6
                expected=residual./sqrt.(sum(abs2,residual;dims=1)./Float32(n).+eps).*weights
                xb=upload(x);bb=upload(branch);wb=upload(weights);out=upload(fill(NaN32,n,tokens))
                g=Mantle.Graph(device)
                Mantle.dispatch!(g,BonsaiRunner.add_rmsnorm_batch_kernel!,
                    (out,xb,bb,wb,Int32(n),eps,Val(Mantle.caps(device).subgroup)),tokens*256;group=256,name="residual_norm_subgroup_regression")
                plan=Mantle.record!(Mantle.Plan(g;budget=Inf));before=Mantle.submissions(device)
                Mantle.run!(plan);Mantle.waitidle(device)
                @test Mantle.submissions(device)-before==1
                @test Array(xb)==residual
                @test all(isfinite,Array(out))
                @test Array(out)≈expected atol=2f-6 rtol=2f-6
            finally
                plan===nothing||Mantle.free!(plan);plan=nothing
                foreach(Mantle.free!,owned);empty!(owned)
            end
        end
    end
end
