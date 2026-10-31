# Trusted fixed infrastructure reproducer ONLY; never pass model code here.
# Run in a fresh root-environment process to distinguish startup/import effects
# from namespace/sandbox restrictions. PoCL CPU backend only.
ENV["JULIA_KA_CPU_THREADS"]="2"
if !isempty(ARGS)
    ARGS[1] in ("none","revise","dnn") || error("unknown trusted import mode")
    ARGS[1]=="revise" && (@eval using Revise)
    ARGS[1]=="dnn" && (@eval using DNNKernels)
end
using KernelAbstractions, Test
@kernel function clean_axpby!(y,x,a,b)
    i=@index(Global,Linear)
    if i<=length(y);@inbounds y[i]=a*x[i]+b*y[i];end
end
function trusted_solve!(y,x,a,b)
    isempty(y) && return y
    backend=KernelAbstractions.CPU()
    clean_axpby!(backend,64)(y,x,a,b;ndrange=length(y))
    KernelAbstractions.synchronize(backend)
    y
end
@testset "trusted CPU startup probe" begin
    for n in (0,1,63,65,129)
        x=Float32.(1:n)./7;y=Float32.(n:-1:1)./11
        expected=2f0.*x .- 0.5f0.*y
        @test trusted_solve!(y,x,2f0,-0.5f0)===y
        @test y≈expected
    end
end
