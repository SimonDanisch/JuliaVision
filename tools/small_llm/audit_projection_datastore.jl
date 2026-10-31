using LinearAlgebra, Statistics, JSON3

function audit_projection_datastore(data)
    keys=Float64.(data.keys);width,records=size(keys)
    means=vec(mean(keys;dims=1));rms=vec(sqrt.(mean(abs2,keys;dims=1)))
    maximum(abs,means)<1e-4 && maximum(abs,rms.-1)<1e-4 || error("datastore normalization invariant")
    centered=keys.-mean(keys;dims=2)
    singular=svdvals(centered);energy=singular.^2
    effective_rank=sum(energy)^2/sum(abs2,energy)
    distances=zeros(Float64,records,records)
    for j in 1:records,i in 1:records
        distances[i,j]=sum(abs2,@view(keys[:,i]).-@view(keys[:,j]))
    end
    neighbors=[argmin([i==j ? Inf : distances[i,j] for i in 1:records]) for j in 1:records]
    report=(;scope="CPU feature-geometry diagnostics, NOT model intelligence/retrieval quality",
        width,records,unique_labels=length(unique(data.labels)),effective_rank,
        maximum_mean_error=maximum(abs,means),maximum_rms_error=maximum(abs,rms.-1),
        nearest_other_distance=(;minimum=minimum(distances[neighbors[j],j] for j in 1:records),
            median=median(distances[neighbors[j],j] for j in 1:records)),
        nearest_other_same_label=count(j->data.labels[neighbors[j]]==data.labels[j],1:records),
        datastore_registration_sha256=data.registration_sha256)
    open(io->JSON3.pretty(io,report),joinpath(@__DIR__,"projection_datastore","geometry_audit.json"),"w");report
end
