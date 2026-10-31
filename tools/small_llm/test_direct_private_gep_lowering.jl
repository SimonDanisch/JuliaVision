include("direct_private_gep_lowering.jl")
using Test,Lava,LLVM
function test_direct_private_gep_lowering()
    base="""
    define i32 @choose(i1 %which, i64 %index, i1 %take) {
    entry:
      %a = alloca [3 x i32]
      %b = alloca [3 x i32]
      store [3 x i32] [i32 11, i32 12, i32 13], ptr %a
      store [3 x i32] [i32 21, i32 22, i32 23], ptr %b
      br i1 %which, label %left, label %right
    left:
      %pa = getelementptr [3 x i32], ptr %a, i64 0, i64 %index
      br label %join
    right:
      %pb = getelementptr [3 x i32], ptr %b, i64 0, i64 %index
      br label %join
    join:
      %p = phi ptr [ %pa, %left ], [ %pb, %right ]
      %v = load i32, ptr %p
      ret i32 %v
    exit:
      ret i32 0
    }
    """
    @testset "Narrow private GEP prototype refuses speculative/clobbered chains" begin
        LLVM.Context() do context
            original=parse(LLVM.Module,base)
            Lava.lower_phi_select_function_ptrs!(original)
            LLVM.verify(original)
            @test occursin("phi ptr",string(original)) # observed current gap, not a hardware miscompile claim
            rewritten=parse(LLVM.Module,base)
            @test DirectPrivateGEPLowering.lower!(rewritten)==1
            LLVM.verify(rewritten)
            @test !occursin("phi ptr",string(rewritten))
            @test occursin("phi i32",string(rewritten))
            @test count("load i32",string(rewritten))==2
            @test DirectPrivateGEPLowering.lower!(rewritten)==0
            conditional=replace(base,"%pa = getelementptr [3 x i32], ptr %a, i64 0, i64 %index\n  br label %join"=>
                "%pa = getelementptr [3 x i32], ptr %a, i64 0, i64 %index\n  br i1 %take, label %join, label %exit")
            clobbered=replace(base,"%v = load"=>"store i32 99, ptr %p\n  %v = load")
            external=replace(base,"%a = alloca [3 x i32]"=>"%a = inttoptr i64 4096 to ptr")
            volatile=replace(base,"%v = load i32"=>"%v = load volatile i32")
            atomic=replace(base,"%v = load i32, ptr %p"=>"%v = load atomic i32, ptr %p monotonic, align 4")
            nested=replace(base,"%v = load i32, ptr %p"=>"store i32 99, ptr %a\n  br label %outer\nouter:\n  %q = phi ptr [ %p, %join ]\n  %v = load i32, ptr %q")
            for source in (conditional,clobbered,external,volatile,atomic,nested)
                mod=parse(LLVM.Module,source)
                @test DirectPrivateGEPLowering.lower!(mod)==0
                LLVM.verify(mod)
                @test occursin("phi ptr",string(mod))
                @test !occursin("direct_gep_edge_load",string(mod))
            end
            unaligned=parse(LLVM.Module,replace(base,"%v = load i32, ptr %p"=>"%v = load i32, ptr %p, align 1"))
            @test DirectPrivateGEPLowering.lower!(unaligned)==1
            LLVM.verify(unaligned)
            @test count("load i32, ptr %p",string(unaligned))==2
            @test count(", align 1",string(unaligned))==2
        end
    end
end
