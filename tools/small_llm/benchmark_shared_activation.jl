using JSON3, Statistics

function benchmark_shared_activation(model;output,tokens=64,repeats=6)
    ispath(output) && error("Refusing to overwrite $output")
    conditions=[("unfused",()),("reuse",(:reuse,)),("automatic",(:reuse,:fuse))]
    sessions=Any[];rows=Any[];reference=nothing
    prompt=[27175,667,17839,958,3145,63368,5638,13]
    try
        for (_,opts) in conditions
            push!(sessions,Qwen35Experiment.session(model;capacity=256,activation_optimizations=opts))
        end
        for trial in 0:repeats
            order=isodd(trial) ? reverse(eachindex(sessions)) : eachindex(sessions)
            for i in order
                s=sessions[i]
                reset_qwen35!(s)
                for token in prompt;Qwen35Experiment.step!(s,token);end
                Mantle.waitidle(model.device)
                before=Mantle.submissions(model.device)
                elapsed=@elapsed begin
                    Qwen35Experiment.decode!(s,tokens);Mantle.waitidle(model.device)
                end
                submissions=Mantle.submissions(model.device)-before
                @assert submissions==tokens
                ids=Array(s.predictions)[length(prompt):length(prompt)+tokens-1]
                logits=Array(s.logits);@assert all(isfinite,logits)
                bits=collect(reinterpret(UInt32,logits))
                if reference===nothing;reference=(ids,bits)
                else;@assert reference==(ids,bits);end
                trial==0 && continue
                push!(rows,(;trial,condition=conditions[i][1],elapsed,tokens_per_second=tokens/elapsed,
                    submissions,passes=length(s.plan.passes),barrier_passes=count(p->!isempty(p.pre),s.plan.passes)))
                println(last(rows));flush(stdout)
                open(io->JSON3.pretty(io,(;complete=false,rows)),output,"w")
            end
        end
        rates=Dict(name=>median([r.tokens_per_second for r in rows if r.condition==name]) for (name,_) in conditions)
        result=(;complete=true,width=Int(model.config.hidden_size),layers=Int(model.config.num_hidden_layers),
            tokens,prompt,rates,tokens_and_logits_bitidentical=true,rows,
            scope="Shared DNNKernels activation representation/reuse and expression-to-Q8; fixed model/weights/core sync")
        open(io->JSON3.pretty(io,result),output,"w");result
    finally
        foreach(close,sessions)
    end
end
