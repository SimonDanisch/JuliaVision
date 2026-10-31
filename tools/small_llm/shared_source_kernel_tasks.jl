isdefined(@__MODULE__,:VULKAN_KERNEL_SETUP) || include("vulkan_kernel_tasks.jl")

"""Prospective authored tasks; no model outputs exist at construction.

Compare interface choices, not hidden-answer injection: direct GPU source vs
ordinary typed serial source consumed by the fixed restricted compiler. Both
must ultimately execute correctly on device arrays. Reference bodies/tests are
evaluator-only and must never enter generation prompts or retrieved packs.
"""
function shared_source_kernel_tasks()
    gpu="Write only a Julia GPU function kernel!, optionally importing KernelInterface as KI. Use the1-based global index KI.get_global_id().x, guard rounded-up invocations, and return nothing. Arguments are device arrays, not host Vector objects. Do not create a device or launcher, use KernelAbstractions, or loop over the whole array. "
    cpu="Write only an ordinary Julia function kernel! with explicit Vector{Float32}/Vector{Int32} array annotations and Int32 scalar annotations. Use exactly one for i=1:N loop and one out[i] assignment per iteration, with optional pure scalar local assignments; return nothing afterward. Do not use macros, branches, nested loops, device APIs, wrappers, output reads or side effects. A fixed compiler maps that loop to unique guarded GPU invocations and changes vector annotations to AbstractVector; it does NOT repair arithmetic. Supported pure operations are +,-,*,/,div,mod,mod1,min,max,clamp,muladd,Float32,Int32,Int. "
    common="All arrays are distinct. Input bounds are guaranteed as specified. Preserve out padding after the valid range. Float32 data except explicitly Int32 index arrays. "
    definitions=[
        (;id="shared_periodic_laplacian",
          specification="Signature kernel!(out,x,n::Int32). n>=0. For1<=i<=n, compute x[mod1(i-1,n)]-2f0*x[i]+x[mod1(i+1,n)]. This is a periodic stencil, NOT endpoint clamping. n=0 does no work; n=1 gives zero. x has at least n elements.",
          reference_cpu="""
          function kernel!(out::Vector{Float32},x::Vector{Float32},n::Int32)
              for i=1:n
                  left=mod1(i-1,n)
                  right=mod1(i+1,n)
                  out[i]=x[left]-2f0*x[i]+x[right]
              end
              nothing
          end
          """,tests=raw"""
          for n in (0,1,2,7,67,259)
              x=Float32[cos(i*.17)+sin(i*.31) for i=1:n+5]
              initial=fill(-789f0,n+5);expected=copy(initial)
              for i=1:n;expected[i]=x[mod1(i-1,n)]-2f0*x[i]+x[mod1(i+1,n)];end
              @test __bench_launch__(kernel!,initial,x,Int32(n);threads=n)≈expected atol=2f-6 rtol=2f-6
          end
          """),
        (;id="shared_indexed_interpolation",
          specification="Signature kernel!(out,x,indices,t,n::Int32,m::Int32). indices is Int32, all other arrays Float32. n>=0, m>=2. For1<=i<=n, let j=clamp(indices[i],1,m-1) and compute (1f0-t[i])*x[j]+t[i]*x[j+1]. t values lie in[0,1]; do not round t or clamp indices to m. x has at least m elements; indices/t have at least n.",
          reference_cpu="""
          function kernel!(out::Vector{Float32},x::Vector{Float32},indices::Vector{Int32},t::Vector{Float32},n::Int32,m::Int32)
              for i=1:n
                  j=clamp(indices[i],1,m-1)
                  out[i]=(1f0-t[i])*x[j]+t[i]*x[j+1]
              end
              nothing
          end
          """,tests=raw"""
          for (n,m) in ((0,2),(1,2),(7,5),(67,13),(259,29))
              x=Float32[sin(i*.19)+i*.07 for i=1:m+3]
              indices=Int32[mod(17i,m+11)-5 for i=1:n+5];t=Float32[mod(7i,13)/12 for i=1:n+5]
              initial=fill(-789f0,n+5);expected=copy(initial)
              for i=1:n
                  j=clamp(indices[i],1,m-1);expected[i]=(1f0-t[i])*x[j]+t[i]*x[j+1]
              end
              @test __bench_launch__(kernel!,initial,x,indices,t,Int32(n),Int32(m);threads=n)≈expected atol=2f-6 rtol=2f-6
          end
          """),
        (;id="shared_column_bias_clip",
          specification="Signature kernel!(out,x,bias,rows::Int32,cols::Int32). rows/cols positive. x/out are flattened COLUMN-MAJOR rows-by-cols matrices. bias has cols entries. For each valid flat index i, let column=div(i-1,rows)+1 and compute clamp(x[i]+bias[column],-1f0,1f0). Bias by COLUMN, not row, and clamp AFTER addition. Valid length N=rows*cols.",
          reference_cpu="""
          function kernel!(out::Vector{Float32},x::Vector{Float32},bias::Vector{Float32},rows::Int32,cols::Int32)
              for i=1:rows*cols
                  column=div(i-1,rows)+1
                  out[i]=clamp(x[i]+bias[column],-1f0,1f0)
              end
              nothing
          end
          """,tests=raw"""
          for (rows,cols) in ((1,5),(3,1),(3,7),(17,11))
              n=rows*cols;x=Float32[1.7sin(i*.21) for i=1:n+5];bias=Float32[cos(i*.7) for i=1:cols]
              initial=fill(-789f0,n+5);expected=copy(initial)
              for i=1:n;expected[i]=clamp(x[i]+bias[div(i-1,rows)+1],-1f0,1f0);end
              @test __bench_launch__(kernel!,initial,x,bias,Int32(rows),Int32(cols);threads=n)≈expected atol=2f-6 rtol=2f-6
          end
          """),
        (;id="shared_block_reverse",
          specification="Signature kernel!(out,x,n::Int32,width::Int32). width positive, n>=0 is a multiple of width. Independently reverse each consecutive width-element block in x: for valid index i, j=div(i-1,width)*width+width-mod(i-1,width), and out[i]=x[j]. Do NOT reverse the entire array or reorder the blocks. x has at least n elements.",
          reference_cpu="""
          function kernel!(out::Vector{Float32},x::Vector{Float32},n::Int32,width::Int32)
              for i=1:n
                  j=div(i-1,width)*width+width-mod(i-1,width)
                  out[i]=x[j]
              end
              nothing
          end
          """,tests=raw"""
          for (n,width) in ((0,7),(1,1),(7,1),(7,7),(65,5),(259,7))
              x=Float32.(1:n+5)./7f0;initial=fill(-789f0,n+5);expected=copy(initial)
              for i=1:n;expected[i]=x[div(i-1,width)*width+width-mod(i-1,width)];end
              @test __bench_launch__(kernel!,initial,x,Int32(n),Int32(width);threads=n)==expected
          end
          """)]
    [(;definition...,category="vulkan_kernel",setup=VULKAN_KERNEL_SETUP,
        gpu_prompt=gpu*common*definition.specification,cpu_prompt=cpu*common*definition.specification) for definition in definitions]
end
