include("score_laptop_native_memory.jl")

function freeze_laptop_native_temperature_study(;directory=joinpath(@__DIR__,"laptop_native_temperature_results"))
    original=joinpath(@__DIR__,"laptop_native_memory_results")
    state=laptop_native_memory_state(original);summary=JSON3.read(read(joinpath(original,"summary.json"),String))
    summary.complete||error("finish all original arms first")
    auditpath=joinpath(@__DIR__,"native_source_geometry_audit.json");audit=JSON3.read(read(auditpath,String))
    audit.datastore_registration_sha256==state.packet.datastore_registration_sha256||error("geometry audit drift")
    geometry=only(filter(r->r.mode=="other_document_only",audit.rows))
    gap=Float64(geometry.median_first_second_gap)
    isfinite(gap)&&gap>0||error("invalid source-only gap")
    temperature=Float32(2.0^round(Int,log2(gap)))
    jobs=[(;id=String(j.id)*"_source_scaled",task=j.task,arm=j.arm,weight=j.weight,shuffled=j.shuffled,messages=j.messages,contract_sha256=j.contract_sha256)
        for j in state.packet.jobs if j.arm in ("memory_10","memory_25","shuffled_25")]
    ispath(directory)&&!isempty(readdir(directory))&&error("refuse overwrite")
    p=state.packet
    packet=(;scope="POSTHOC source-only temperature calibration after a negative tiny native-memory pilot; same two already-used development tasks, NOT held-out evidence",
        design="Same six nonzero/shuffled interventions, but temperature is nearest power of two to median nearest-neighbor distance GAP across the TWO source documents. No task-answer/score/seed sweep. Original zero/source-context controls retained, not regenerated.",
        policy=p.policy,jobs,budget_tokens=p.budget_tokens,thinking=p.thinking,temperature,topk=p.topk,shuffle_seed=p.shuffle_seed,
        datastore_registration_sha256=p.datastore_registration_sha256,records=p.records,width=p.width,
        parent_conditions_sha256=bytes2hex(open(sha256,state.path)),parent_summary_sha256=bytes2hex(open(sha256,joinpath(original,"summary.json"))),
        geometry_audit_sha256=bytes2hex(open(sha256,auditpath)),source_only_median_gap=gap,
        source_sha256=merge(Dict(String(k)=>String(v) for (k,v) in pairs(p.source_sha256)),
            Dict("laptop_native_temperature_study.jl"=>bytes2hex(open(sha256,@__FILE__)))))
    mkpath(directory);open(io->JSON3.pretty(io,packet),joinpath(directory,"conditions.json"),"w");packet
end
