const VULKAN_KERNEL_SETUP="""
using Mantle
import KernelInterface as KI
import Atomix
@assert !ispath("/dev/dri")
__bench_device=Mantle.Device("llvmpipe")
@assert occursin("llvmpipe",lowercase(string(Mantle.devicename(__bench_device))))
function __bench_launch__(kernel,initial,args...;threads)
    owned=Any[];plan=nothing
    upload(x)=x isa AbstractArray ? begin b=Mantle.Buffer(__bench_device,x);push!(owned,b);b;end : x
    try
        output=upload(initial);parameters=map(upload,args)
        graph=Mantle.Graph(__bench_device)
        Mantle.dispatch!(graph,kernel,(output,parameters...),64*cld(max(threads,1),64);group=64)
        plan=Mantle.record!(Mantle.Plan(graph;budget=Inf));Mantle.run!(plan)
        result=Array(output);Mantle.waitidle(__bench_device);result
    finally
        plan===nothing || Mantle.free!(plan)
        foreach(Mantle.free!,reverse(owned))
    end
end
"""

function vulkan_kernel_tasks()
    prefix="Write ONLY the Julia kernel function kernel! specified below, optionally including imports. Use KernelInterface as KI: a global1D index is KI.get_global_id().x. Do NOT create devices, compile/launch kernels, use KernelAbstractions @kernel, or write host wrappers. Arguments are device arrays/scalars and indices are1-based. Return nothing. "
    [
    (;id="ki_body_affine",category="vulkan_kernel",setup=VULKAN_KERNEL_SETUP,
      prompt=prefix*"Signature kernel!(out,x,alpha::Float32,beta::Float32,n::Int32). Write out[i]=muladd(alpha,x[i],beta) for1<=i<=n. Launches round up; leave all out entries after n unchanged, including when n=0. All input/output storage is distinct and may contain padding. Float32 arithmetic.",
      reference="""
      function kernel!(out,x,alpha::Float32,beta::Float32,n::Int32)
          i=Int32(KI.get_global_id().x)
          i<=n && (@inbounds out[i]=muladd(alpha,x[i],beta))
          nothing
      end
      """,tests="""
      for n in (0,1,7,65,257)
          x=Float32[sin(i*.07) for i in 1:n+5];initial=fill(-1234f0,n+5);expected=copy(initial)
          expected[1:n].=muladd.(1.75f0,x[1:n],-0.25f0)
          actual=__bench_launch__(kernel!,initial,x,1.75f0,-0.25f0,Int32(n);threads=n)
          @test actual ≈ expected atol=1f-6 rtol=1f-6
      end
      """),
    (;id="ki_body_clamped_stencil",category="vulkan_kernel",setup=VULKAN_KERNEL_SETUP,
      prompt=prefix*"Signature kernel!(out,x,n::Int32). For1<=i<=n write (x[max(i-1,1)]+x[i]+x[min(i+1,n)])/3f0. The stencil CLAMPS neighbor indices to the first/last valid input; do not simply copy endpoints. n may be0 or1. Buffers are distinct; leave padded out entries after n unchanged. Float32 arithmetic.",
      reference="""
      function kernel!(out,x,n::Int32)
          i=Int32(KI.get_global_id().x)
          if i<=n
              @inbounds out[i]=(x[max(i-Int32(1),Int32(1))]+x[i]+x[min(i+Int32(1),n)])/3f0
          end
          nothing
      end
      """,tests="""
      for n in (0,1,2,67,259)
          x=Float32[cos(i*.13) for i in 1:n+5];initial=fill(-1234f0,n+5);expected=copy(initial)
          for i in 1:n;expected[i]=(x[max(i-1,1)]+x[i]+x[min(i+1,n)])/3f0;end
          @test __bench_launch__(kernel!,initial,x,Int32(n);threads=n) ≈ expected atol=1f-6 rtol=1f-6
      end
      """),
    (;id="ki_body_column_major_scale",category="vulkan_kernel",setup=VULKAN_KERNEL_SETUP,
      prompt=prefix*"Signature kernel!(out,x,scales,rows::Int32,cols::Int32). x and out are FLATTENED Julia column-major matrices. For each valid matrix entry multiply x by scales[row], NOT scales[column]. rows and cols are positive. Launch rounds up; leave out padding after rows*cols unchanged. Buffers are distinct and Float32. Use a1D global index.",
      reference="""
      function kernel!(out,x,scales,rows::Int32,cols::Int32)
          i=Int32(KI.get_global_id().x)
          if i<=rows*cols
              row=(i-Int32(1))%rows+Int32(1)
              @inbounds out[i]=x[i]*scales[row]
          end
          nothing
      end
      """,tests="""
      for (rows,cols) in ((1,1),(3,5),(7,13),(17,11))
          n=rows*cols;x=Float32.(1:n+5)./7f0;scales=Float32.(1:rows).*0.25f0
          initial=fill(-1234f0,n+5);expected=copy(initial)
          for i in 1:n;expected[i]=x[i]*scales[mod1(i,rows)];end
          @test __bench_launch__(kernel!,initial,x,scales,Int32(rows),Int32(cols);threads=n) ≈ expected
      end
      """),
    (;id="ki_body_atomic_max",category="vulkan_kernel",setup=VULKAN_KERNEL_SETUP,
      prompt=prefix*"Signature kernel!(out,labels,x,n::Int32). For every1<=i<=n atomically update out[labels[i]] to the maximum of its old value and x[i]. labels are valid1-based Int32 indices. x/out are Float32, out initially -Inf. Many workgroups contend for the same outputs: a normal load followed by an atomic STORE is NOT a valid maximum RMW. Use current Atomix.jl with explicit :monotonic ordering: only this independent reduction result is read after kernel completion. Ignore rounded-up threads past n. No scratch arrays or host reductions.",
      reference="""
      function kernel!(out,labels,x,n::Int32)
          i=Int32(KI.get_global_id().x)
          if i<=n
              @inbounds Atomix.@atomic :monotonic max(out[labels[i]],x[i])
          end
          nothing
      end
      """,tests="""
      for n in (1,257,4096), repetition in 1:3
          labels=Int32[mod(i*3,7)+1 for i in 1:n]
          x=Float32[mod(i*1543+repetition*719,8191)-4096 for i in 1:n]
          initial=fill(-Inf32,7);expected=copy(initial)
          for i in 1:n;expected[labels[i]]=max(expected[labels[i]],x[i]);end
          @test __bench_launch__(kernel!,initial,labels,x,Int32(n);threads=n)==expected
      end
      """)
    ]
end
