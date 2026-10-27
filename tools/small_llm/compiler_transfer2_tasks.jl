isdefined(@__MODULE__,:VULKAN_KERNEL_SETUP) || include("vulkan_kernel_tasks.jl")

"""Fresh authored arithmetic/indexing tasks, reusing ONLY public type interfaces.
Reference values and tests are evaluator-only and never enter model prompts.
"""
function compiler_transfer2_tasks()
    definitions=[
        (;id="transfer2_backward_gradient",interface_id="shared_periodic_laplacian",
            specification="Signature kernel!(out,x,n::Int32). n>=0. For valid index i compute (3f0*x[i]-4f0*x[mod1(i-1,n)]+x[mod1(i-2,n)])/2f0. Periodic BACKWARD stencil, not the symmetric Laplacian. n=0 does no work; x has at least n elements.",
            reference_value="(3f0*x[i]-4f0*x[mod1(i-1,n)]+x[mod1(i-2,n)])/2f0",
            tests=raw"""
            for n in (0,1,2,9,71,263)
                x=Float32[sin(i*.23)+.03i for i=1:n+5];initial=fill(-913f0,n+5);expected=copy(initial)
                if n>0
                    values=x[1:n]
                    expected[1:n].=(3f0.*values.-4f0.*circshift(values,1).+circshift(values,2))./2f0
                end
                @test __bench_launch__(kernel!,initial,x,Int32(n);threads=n)≈expected atol=2f-6 rtol=2f-6
            end
            """),
        (;id="transfer2_weighted_forward_drop",interface_id="shared_indexed_interpolation",
            specification="Signature kernel!(out,x,indices,t,n::Int32,m::Int32). indices is Int32, other arrays Float32. n>=0,m>=1. Let j=clamp(indices[i],1,m). Compute t[i]*(x[j]-x[min(j+1,m)]). This is a signed weighted forward DROP, NOT interpolation. t may be negative. x has at least m elements, indices/t at least n.",
            reference_value="t[i]*(x[clamp(indices[i],1,m)]-x[min(clamp(indices[i],1,m)+1,m)])",
            tests=raw"""
            for (n,m) in ((0,1),(1,1),(9,2),(71,17),(263,31))
                x=Float32[cos(i*.31)+.07i for i=1:m+4]
                indices=Int32[mod(19i,m+9)-4 for i=1:n+5];t=Float32[mod(11i,17)/8-1 for i=1:n+5]
                initial=fill(-913f0,n+5);expected=copy(initial)
                for i=1:n
                    j=max(1,min(Int(indices[i]),m));next=j==m ? j : j+1
                    expected[i]=(x[j]-x[next])*t[i]
                end
                @test __bench_launch__(kernel!,initial,x,indices,t,Int32(n),Int32(m);threads=n)≈expected atol=2f-6 rtol=2f-6
            end
            """),
        (;id="transfer2_column_difference_squares",interface_id="shared_column_bias_clip",
            specification="Signature kernel!(out,x,bias,rows::Int32,cols::Int32). rows/cols positive. Arrays Float32, flattened COLUMN-MAJOR rows-by-cols. For i let c=div(i-1,rows)+1 and compute (x[i]-bias[c])*(x[i]+bias[c]). Do NOT clamp, add a row bias or square the difference. Valid length rows*cols; bias has cols entries.",
            reference_value="(x[i]-bias[div(i-1,rows)+1])*(x[i]+bias[div(i-1,rows)+1])",
            tests=raw"""
            for (rows,cols) in ((1,7),(5,1),(5,9),(19,13))
                n=rows*cols;x=Float32[1.7sin(i*.17) for i=1:n+5];bias=Float32[cos(i*.41) for i=1:cols]
                initial=fill(-913f0,n+5);expected=copy(initial)
                for c=1:cols,r=1:rows
                    i=r+(c-1)*rows;expected[i]=x[i]*x[i]-bias[c]*bias[c]
                end
                @test __bench_launch__(kernel!,initial,x,bias,Int32(rows),Int32(cols);threads=n)≈expected atol=2f-6 rtol=2f-6
            end
            """),
        (;id="transfer2_block_left_rotate",interface_id="shared_block_reverse",
            specification="Signature kernel!(out,x,n::Int32,width::Int32). width positive, n>=0 multiple of width. Rotate EACH consecutive width-element block LEFT by ONE element, wrapping its first element to its own end. Do NOT reverse blocks or rotate the whole array. Valid length n; x has at least n entries.",
            reference_value="x[div(i-1,width)*width+mod(i,width)+1]",
            tests=raw"""
            for (n,width) in ((0,9),(1,1),(9,1),(9,9),(65,5),(259,7))
                x=Float32.(1:n+5)./11f0;initial=fill(-913f0,n+5);expected=copy(initial)
                for start=1:width:n
                    block=x[start:start+width-1];expected[start:start+width-1]=vcat(block[2:end],block[1:1])
                end
                @test __bench_launch__(kernel!,initial,x,Int32(n),Int32(width);threads=n)==expected
            end
            """)]
    [(;d...,category="vulkan_kernel",setup=VULKAN_KERNEL_SETUP) for d in definitions]
end
