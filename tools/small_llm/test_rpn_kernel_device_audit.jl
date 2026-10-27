include("rpn_kernel_device_audit.jl")
using Test

function test_rpn_kernel_device_audit(device)
    @testset "Grading-only RPN CPU/device math + wrong-arithmetic rejection" begin
        cases=[let x=Float32[-1.25,0.25,2.5][1:max(n,1)],initial=fill(-913f0,n+7)
            # Constant-only public interface illustration, not a stencil answer.
            expected=copy(initial);expected[1:n].=0.5f0
            (;n,x,initial,expected)
        end for n in (0,1,3)]
        good=audit_rpn_kernel_device(device,"two\none\nsub\nhalf\nmul\nfinish",cases)
        @test good.admitted&&good.passed
        @test length(good.rows)==3
        @test all(r->r.cpu_match&&r.gpu_match&&r.padding&&r.submissions==1,good.rows)
        wrong=audit_rpn_kernel_device(device,"two\none\nadd\nhalf\nmul\nfinish",cases)
        @test wrong.admitted&&!wrong.passed
        @test any(r->!r.cpu_match&&!r.gpu_match,wrong.rows)
        rejected=audit_rpn_kernel_device(device,"run(`touch /tmp/not-allowed`)\nfinish",cases)
        @test !rejected.admitted&&!rejected.passed&&isempty(rejected.rows)
        @test_throws ArgumentError audit_rpn_kernel_device(device,"x0\nfinish",Any[])
        invalid=[(;n=3,x=Float32[1],initial=zeros(Float32,3),expected=zeros(Float32,3))]
        @test_throws ArgumentError audit_rpn_kernel_device(device,"x0\nfinish",invalid)
    end
end
