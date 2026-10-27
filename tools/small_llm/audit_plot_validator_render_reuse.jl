using JSON3,SHA
function plot_validator_render_reuse_decision(row,prior)
    !row.admitted&&return "unsupported_output"
    !row.retained_original&&return "pending_new_plan_render"
    prior!==nothing&&prior.id==row.selected_generation_id&&prior.generation_sha256==row.generation_sha256||error("Original render identity drift")
    prior.rendered ? prior.within_canvas ? "reused_original_unclipped" : "reused_original_clipped" : "original_render_failed"
end

function audit_plot_validator_render_reuse(;directory=joinpath(@__DIR__,"plot_validator_results"),parent=joinpath(@__DIR__,"plot_interface_results"))
    digest(path)=bytes2hex(sha256(read(path)))
    primarypath=joinpath(directory,"summary.json");primary=JSON3.read(read(primarypath,String))
    originalpath=joinpath(parent,"render_audit_summary.json");original=JSON3.read(read(originalpath,String))
    primary.complete&&length(primary.rows)==16&&original.complete||error("Complete primary/original render audit required")
    original.primary_summary_sha256==digest(joinpath(parent,"summary.json"))||error("Original render source drift")
    cached=Dict(String(r.id)=>r for r in original.rows)
    rows=map(primary.rows) do row
        id=String(row.selected_generation_id);basename(id)==id||error("Safe selected ID required")
        prior=get(cached,id,nothing);status=plot_validator_render_reuse_decision(row,prior)
        image=nothing
        if startswith(status,"reused_original_")
            image=joinpath(parent,"render_artifacts",id*".png")
            isfile(image)&&!islink(image)&&filesize(image)<=8_000_000&&digest(image)==prior.render.sha256||error("Original image drift")
            digest(joinpath(parent,id*".json"))==row.generation_sha256||error("Original generation drift")
        end
        (;id=row.id,arm=row.arm,status,selected_generation_id=id,artifact=image===nothing ? nothing : abspath(image),
            generation_sha256=row.generation_sha256,rendered=startswith(status,"reused_original_"),aesthetic_score=nothing)
    end
    result=(;complete=all(r->r.status!="pending_new_plan_render",rows),rows,
        scope="POSTHOC exact-byte original-image provenance reuse ONLY, not new renders, independent trials or aesthetic evaluation. New admitted revisions would remain PENDING until separately rendered.",
        unique_reused_images=length(unique(r.artifact for r in rows if r.rendered)),
        primary_summary_sha256=digest(primarypath),original_render_summary_sha256=digest(originalpath),source_sha256=digest(@__FILE__))
    open(io->JSON3.pretty(io,result),joinpath(directory,"render_reuse_audit.json"),"w");result
end
