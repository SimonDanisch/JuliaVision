using JSON3

function public_example_output_names(jobs)
    names=String[String(j.id) for j in jobs]
    isempty(names)&&throw(ArgumentError("Empty collection cohort"))
    length(unique(names))==length(names)||throw(ArgumentError("Duplicate jobs"))
    all(id->!isempty(id)&&basename(id)==id&&!occursin('\0',id)&&!occursin('\\',id)&&!(id in (".","..")),names)||
        throw(ArgumentError("Unsafe job identifier"))
    names
end

function save_immutable_example_outputs(directory,files,allowed_names)
    allowed=Set(vcat([id*".json" for id in allowed_names],[id*"_attempt.json" for id in allowed_names]))
    all(name->name in allowed,keys(files))||error("Unexpected remote output")
    for id in allowed_names
        output=id*".json";attempt=id*"_attempt.json"
        haskey(files,output)==haskey(files,attempt)||error("Unpaired output/attempt")
    end
    # Validate every existing file BEFORE writing anything new.
    for (name,data) in files
        path=joinpath(directory,name);islink(path)&&error("Linked local output")
        isfile(path)&&read(path)!=data&&error("Immutable output changed")
    end
    for (name,data) in files
        path=joinpath(directory,name)
        isfile(path)||open(io->write(io,data),path,"w")
    end
    length(files)÷2
end

"""Read-only remote collection: no package dependencies in the file-only worker.

Only public job identifiers cross to that worker. Never upload ROOT score files,
private formulas, probes, references or complete local result directories.
"""
function collect_public_example_outputs(file_session,directory)
    p=JSON3.read(read(joinpath(directory,"conditions.json"),String))
    names=public_example_output_names(p.jobs)
    files=file_session(names) do ids
        root="/tmp/small-llm-bosgame-20261025/dev/JuliaVision/tools/small_llm/rpn_public_example_results"
        islink(root)&&error("Linked remote output directory")
        found=Dict{String,Vector{UInt8}}()
        for id in ids
            basename(id)==id||error("Unsafe remote identifier")
            output=id*".json";attempt=id*"_attempt.json"
            if isfile(joinpath(root,output))
                isfile(joinpath(root,attempt))||error("Missing attempt receipt")
                (islink(joinpath(root,output))||islink(joinpath(root,attempt)))&&error("Linked remote output")
                found[output]=read(joinpath(root,output));found[attempt]=read(joinpath(root,attempt))
            end
        end
        found
    end
    save_immutable_example_outputs(directory,files,names)
end
