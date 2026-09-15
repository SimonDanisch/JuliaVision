# Isolate the replay cost, not Julia's immediate launch/compile overhead.
# Include in bt_julia_eval; no model weights are loaded by this file.
using DNNKernels, Mantle, KernelAbstractions, Statistics, Test, Random

function bench_masked_flash(backend; lq=4096, lk=512,
        tiles=((16,32,8),(32,32,8),(32,64,8),(16,64,8),(32,32,4)),
        pads=((8,0),(16,0),(8,8),(16,8)))
    rng=MersenneTwister(73)
    caps=DNNKernels.caps(backend)
    q=DNNKernels.toback(backend,randn(rng,Float16,128,lq,8,1).*Float16(.1))
    k=DNNKernels.toback(backend,randn(rng,Float16,128,lk,8,1).*Float16(.1))
    v=DNNKernels.toback(backend,randn(rng,Float16,128,lk,8,1).*Float16(.1))
    out=similar(q)
    mh=[s<=mod(r-1,max(1,lq÷8))+1 ? Float16(0) : -floatmax(Float16)
        for s in 1:lk,r in 1:lq,h in 1:1,b in 1:1]
    mask=DNNKernels.toback(backend,mh)
    g=DNNKernels.Graph("empty",String[],String[],String[],
        Dict{String,DNNKernels.Buffer}(),String[],DNNKernels.Op[])
    ctx=DNNKernels.Ctx(Dict{String,Any}(),g,(;),backend;
        ws=DNNKernels.Workspace(backend))
    results=[]; ref=nothing
    for (br,bc,nw) in tiles, (epad,rpad) in pads
        DNNKernels.flashcmshared(128,br,bc,epad,rpad)<=caps.sharedbudget || continue
        p=DNNKernels.flashcm_plan(caps,q,k,v,nothing;
            clamp=true,BR=br,BC=bc,NW=nw)
        p isa DNNKernels.Decline && continue
        run()=begin
            DNNKernels.reset!(ctx.ws)
            DNNKernels.sdpaflashcm!(ctx,out,p,q,k,v,.08838835f0;mask,epad,rpad)
            nothing
        end
        run(); KernelAbstractions.synchronize(backend); got=Array(out)
        if ref===nothing
            ref=got
        else
            @test maximum(abs,Float32.(got).-Float32.(ref))<.001
        end
        mg=Mantle.Graph(Mantle.Device(backend))
        Mantle.record_into(run,mg,"attention")
        pl=Mantle.record!(Mantle.Plan(mg))
        times=Float64[]
        for _ in 1:5
            push!(times,1000*@elapsed(begin
                for _ in 1:3; Mantle.run!(pl); end
                KernelAbstractions.synchronize(backend)
            end)/3)
        end
        push!(results,(median(times),br,bc,nw,epad,rpad))
        Mantle.free!(pl)
    end
    if lq%128==0 && lk%128==0 && 128<=lk<=1024
        runstaged()=begin
            DNNKernels.reset!(ctx.ws)
            DNNKernels.maskedprefill!(ctx,out,q,k,v,mask,.08838835f0)
            nothing
        end
        runstaged(); KernelAbstractions.synchronize(backend)
        @test maximum(abs,Float32.(Array(out)).-Float32.(ref))<.001
        mg=Mantle.Graph(Mantle.Device(backend))
        Mantle.record_into(runstaged,mg,"staged_attention")
        pl=Mantle.record!(Mantle.Plan(mg))
        times=Float64[]
        for _ in 1:7
            push!(times,1000*@elapsed(begin
                for _ in 1:3; Mantle.run!(pl); end
                KernelAbstractions.synchronize(backend)
            end)/3)
        end
        println("STAGED attention lq=$lq lk=$lk ms=$(median(times))")
        Mantle.free!(pl)
    end
    sort!(results); foreach(println,results)
    results
end
