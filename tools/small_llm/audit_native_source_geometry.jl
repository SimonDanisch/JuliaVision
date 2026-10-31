include("laptop_native_memory_study.jl")
using Statistics,LinearAlgebra

"""Source-only retrieval diagnostic. No benchmark answers or weight tuning."""
function audit_native_source_geometry()
    data=load_laptop_native_data();p=laptop_native_datastore_state(joinpath(@__DIR__,LAPTOP_NATIVE_DIRECTORY)).packet
    keys=Float64.(data.keys);labels=data.labels;n=length(labels)
    norms=vec(sum(abs2,keys;dims=1))
    distances=max.(norms .+ transpose(norms) .- 2 .* (transpose(keys)*keys),0.0)
    owners=reduce(vcat,[fill(String(r.id),length(r.token_ids)-1) for r in p.rows])
    length(owners)==n||error("source boundaries")
    rows=Any[]
    for mode in ("other_tokens_in_same_corpus","other_document_only")
        correct=0;covered=0;masses=Float64[];gaps=Float64[];nearest=Float64[]
        for i=1:n
            candidates=[j for j=1:n if j!=i && (mode!="other_document_only"||owners[j]!=owners[i])]
            sort!(candidates;by=j->(distances[j,i],j));neighbors=first(candidates,min(8,length(candidates)))
            isempty(neighbors)&&error("empty retrieval fixture")
            correct+=labels[first(neighbors)]==labels[i]
            covered+=any(j->labels[j]==labels[i],candidates)
            d=distances[neighbors,i];weights=exp.(-(d.-first(d))./32.0)
            push!(masses,sum(weights[k] for k in eachindex(neighbors) if labels[neighbors[k]]==labels[i];init=0.0)/sum(weights))
            push!(nearest,first(d));length(d)>1&&push!(gaps,d[2]-d[1])
        end
        push!(rows,(;mode,queries=n,top1_correct=correct,top1_accuracy=correct/n,
            labels_available_elsewhere=covered,mean_target_mass_top8_temperature32=mean(masses),
            median_nearest_squared_L2=median(nearest),median_first_second_gap=median(gaps)))
    end
    packet=(;scope="POSTHOC source-token geometry diagnostic only. Even leaving out a token is not held-out language modeling: neighboring records share the teacher-forced source. Cross-document results concern only TWO tiny API sources. No model-task answers, hyperparameter search or quality claim.",
        records=n,width=data.width,distinct_labels=length(unique(labels)),vocabulary=data.vocabulary,
        vocabulary_label_coverage=length(unique(labels))/data.vocabulary,hidden_norm_range=extrema(sqrt.(norms)),rows,
        datastore_registration_sha256=data.registration_sha256,source_sha256=bytes2hex(open(sha256,@__FILE__)))
    open(io->JSON3.pretty(io,packet),joinpath(@__DIR__,"native_source_geometry_audit.json"),"w");packet
end
