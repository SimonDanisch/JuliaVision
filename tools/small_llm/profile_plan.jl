# Where a recorded plan's time goes: every compute pass re-recorded alone on the
# plan's own storage, and the whole chain run in plan order, so each pass reads
# what it would read inside the plan (transients alias; a pass run out of order
# reads another pass's data). Each pass's median GPU timestamp interval over
# `reps` chains, so the submission and the wait are not in the number, summed by
# a name category.
using Printf
import KernelInterface as KI

passcategory(name)=replace(name,r"^layers\.\d+\."=>"",r"\.\d+$"=>"")

function readfirst!(dst,src)
    KI.get_global_id().x==1 && (@inbounds dst[1]=src[1])
    nothing
end

# A plan that reads every GPURef the profiled plan has host-written, so a value
# `reset()` stores lands in the GPURef's device store. A semantic graph's passes
# take a GPURef as a view of that store, so their replays carry no update pass
# of their own and would read what the last full run left there.
function landplan(device,plan)
    refs=[h for h in plan.hostwritten if h isa Mantle.GPURef]
    isempty(refs) && return nothing
    g=Mantle.Graph(device)
    for (k,r) in enumerate(refs)
        scratch=Mantle.Transient.Buffer(g,eltype(Mantle.storage(r)),(1,))
        Mantle.dispatch!(g,readfirst!,(scratch,r),1;group=1,name="land.$k")
    end
    Mantle.record!(Mantle.Plan(g;budget=Inf))
end

function passplans(device,plan;skip=())
    out=Tuple{String,Any}[]
    for pp in plan.passes
        pp.pass.kind===:compute || continue
        name=pp.pass.name;any(s->occursin(s,name),skip) && continue
        g=Mantle.Graph(device)
        for d in pp.pass.dispatches
            # A GPURef stays itself: its value reaches the device through the
            # plan's update pass, and its bare storage holds whatever the last
            # plan that had one wrote (a decode position, a chunk start).
            args=map(a->a isa Mantle.GPURef ? a : Mantle.storage(a),d.args)
            Mantle.dispatch!(g,d.kernel,args,d.ndrange;group=d.group,name)
        end
        push!(out,(name,Mantle.record!(Mantle.Plan(g;budget=Inf,profile=true))))
    end
    out
end

"""Seconds per pass category, the median GPU time over `reps` in-order chains.
`reset()` runs before every chain (e.g. restoring a position the plan advances).
Also prints the `show` slowest single passes whose names contain `detail`."""
function profile_plan(device,plan;reps=5,skip=(),reset=()->nothing,top=25,detail=nothing,show=10)
    plans=passplans(device,plan;skip);land=landplan(device,plan)
    best=zeros(length(plans))
    try
        for _ in 1:reps
            reset()
            land===nothing||(Mantle.run!(land);Mantle.waitidle(device))
            for (_,p) in plans
                Mantle.run!(p);Mantle.waitidle(device)
            end
        end
        best.=[only(t for t in Mantle.timings(p) if t.kind===:compute).gpu_ms/1000 for (_,p) in plans]
    finally
        foreach(p->Mantle.free!(p[2]),plans)
        land===nothing||Mantle.free!(land)
    end
    agg=Dict{String,Tuple{Int,Float64}}()
    for ((name,_),t) in zip(plans,best)
        k=passcategory(name);n,s=get(agg,k,(0,0.0));agg[k]=(n+1,s+t)
    end
    total=sum(best)
    @printf("%d passes, %.2f ms summed\n",length(plans),1000total)
    for (k,(n,s)) in first(sort(collect(agg);by=x->-x[2][2]),top)
        @printf("  %-40s n=%-5d %8.3f ms  %5.1f%%\n",first(k,40),n,1000s,100s/total)
    end
    if detail!==nothing
        picked=[(name,t) for ((name,_),t) in zip(plans,best) if occursin(detail,name)]
        for (name,t) in first(sort(picked;by=x->-x[2]),show)
            @printf("    %-48s %8.3f ms\n",first(name,48),1000t)
        end
    end
    agg
end
