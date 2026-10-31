isdefined(@__MODULE__,:VULKAN_KERNEL_SETUP) || include("vulkan_kernel_tasks.jl")

# Prospective authored validation after the first four-task development study.
# No model output for these tasks has been generated at task construction.
function kernel_transfer_tasks()
    prefix="Write ONLY a Julia GPU function kernel! (optional imports), using KernelInterface as KI and its 1-based global 1D index KI.get_global_id().x. Arguments are device arrays/scalars, NOT ordinary host Vector objects. No host wrappers/devices, KernelAbstractions @kernel, CPU loops over the whole input, or host reductions. Work is rounded up; guard inactive invocations. End the function in return nothing. "
    [
    (;id="transfer_clamped_gather",category="vulkan_kernel",setup=VULKAN_KERNEL_SETUP,
      prompt=prefix*"Signature kernel!(out,x,indices,n::Int32,m::Int32). out and x are distinct Float32 device arrays; indices is a separate Int32 device array. For each1<=i<=n, write x[clamp(indices[i],1,m)] to out[i]. m is positive and x has at least m entries; n may be0. Preserve out padding past n. Do not assume indices already lie in1:m.",
      reference="""
      function kernel!(out,x,indices,n::Int32,m::Int32)
          i=Int32(KI.get_global_id().x)
          if i<=n
              @inbounds out[i]=x[clamp(indices[i],Int32(1),m)]
          end
          nothing
      end
      """,tests="""
      for (n,m) in ((0,1),(1,3),(7,4),(67,13),(257,19))
          x=Float32[cos(i*.19) for i in 1:m+3]
          indices=Int32[mod(i*11,m+9)-4 for i in 1:n+3]
          initial=fill(-789f0,n+3);expected=copy(initial)
          for i in 1:n;expected[i]=x[clamp(indices[i],1,m)];end
          @test __bench_launch__(kernel!,initial,x,indices,Int32(n),Int32(m);threads=n)==expected
      end
      """),
    (;id="transfer_sincos_pair",category="vulkan_kernel",setup=VULKAN_KERNEL_SETUP,
      prompt=prefix*"Signature kernel!(out_sin,out_cos,x,n::Int32). Three distinct Float32 device arrays. For1<=i<=n, write sin(x[i]) into out_sin[i] and cos(x[i]) into out_cos[i]. x values lie in[-3,3]. n can be0; preserve padding in BOTH output arrays after n. Compute each valid element once, no atomics.",
      reference="""
      function kernel!(out_sin,out_cos,x,n::Int32)
          i=Int32(KI.get_global_id().x)
          if i<=n
              @inbounds begin
                  out_sin[i]=sin(x[i]);out_cos[i]=cos(x[i])
              end
          end
          nothing
      end
      """,tests="""
      for n in (0,1,9,65,259)
          x=Float32[3sin(i*.21) for i in 1:n+3];initial=fill(-789f0,n+3)
          # The common helper uploads all array arguments and frees them. Here
          # use two independent invocations, one observing each output position.
          function observe_pair(swapped)
              device=__bench_device;buffers=Any[];plan=nothing
              try
                  s=Mantle.Buffer(device,initial);push!(buffers,s)
                  c=Mantle.Buffer(device,initial);push!(buffers,c)
                  input=Mantle.Buffer(device,x);push!(buffers,input)
                  g=Mantle.Graph(device)
                  Mantle.dispatch!(g,kernel!,(s,c,input,Int32(n)),64*cld(max(n,1),64);group=64)
                  plan=Mantle.record!(Mantle.Plan(g;budget=Inf));Mantle.run!(plan);Mantle.waitidle(device)
                  swapped ? Array(c) : Array(s)
              finally
                  plan===nothing || Mantle.free!(plan);foreach(Mantle.free!,reverse(buffers))
              end
          end
          es=copy(initial);ec=copy(initial);es[1:n].=sin.(x[1:n]);ec[1:n].=cos.(x[1:n])
          @test observe_pair(false) ≈ es atol=2f-6 rtol=2f-6
          @test observe_pair(true) ≈ ec atol=2f-6 rtol=2f-6
      end
      """),
    (;id="transfer_flat_transpose",category="vulkan_kernel",setup=VULKAN_KERNEL_SETUP,
      prompt=prefix*"Signature kernel!(out,x,rows::Int32,cols::Int32). x is a FLATTENED column-major rows-by-cols Float32 matrix; out is distinct and must hold its transposed cols-by-rows matrix, also flattened COLUMN-MAJOR. rows/cols are positive. Exactly rows*cols valid outputs, followed by padding that must stay unchanged. Handle rectangular matrices, not just squares; no scratch memory.",
      reference="""
      function kernel!(out,x,rows::Int32,cols::Int32)
          i=Int32(KI.get_global_id().x)
          if i<=rows*cols
              r=(i-Int32(1))%cols;c=(i-Int32(1))÷cols
              @inbounds out[i]=x[c+r*rows+Int32(1)]
          end
          nothing
      end
      """,tests="""
      for (rows,cols) in ((1,5),(3,1),(3,7),(17,11))
          n=rows*cols;x=Float32.(1:n+3)./7f0;initial=fill(-789f0,n+3);expected=copy(initial)
          expected[1:n].=vec(permutedims(reshape(x[1:n],rows,cols)))
          @test __bench_launch__(kernel!,initial,x,Int32(rows),Int32(cols);threads=n)==expected
      end
      """),
    (;id="transfer_atomic_sum",category="vulkan_kernel",setup=VULKAN_KERNEL_SETUP,
      prompt=prefix*"Signature kernel!(out,labels,x,n::Int32). out/x Float32, labels Int32; all storage is distinct. For every1<=i<=n atomically ADD x[i] into out[labels[i]], retaining out's initial values. labels are valid1-based indices, and many workgroups contend. n can be0. Use Atomix.jl with explicit :monotonic ordering; results are observed only after kernel completion. A separate normal load followed by an atomic STORE is not a valid RMW. No scratch arrays or host reductions.",
      reference="""
      function kernel!(out,labels,x,n::Int32)
          i=Int32(KI.get_global_id().x)
          if i<=n
              @inbounds Atomix.@atomic :monotonic out[labels[i]] += x[i]
          end
          nothing
      end
      """,tests="""
      for n in (0,1,257,4096), trial in 1:2
          labels=Int32[mod(i*3,7)+1 for i in 1:n]
          x=Float32[mod(i*13+trial*7,11)-5 for i in 1:n]./8f0
          initial=Float32[.5,-1,.25,2,-.5,1.25,-2];expected=copy(initial)
          for i in 1:n;expected[labels[i]]+=x[i];end
          # Bounded dyadic inputs keep every possible partial sum exactly
          # representable: bitwise comparison does not assume one atomic order.
          @test __bench_launch__(kernel!,initial,labels,x,Int32(n);threads=n)==expected
      end
      """)]
end
