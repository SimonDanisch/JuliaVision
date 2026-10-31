using TOML,SHA,JSON3,Pkg

"""Build a portable, pinned inference snapshot; never edit either live project."""
function prepare_laptop_bundle(root=abspath(joinpath(@__DIR__,"../../../.."));include_native_model=true)
    root=abspath(root)
    manifest=TOML.parsefile(joinpath(root,"Manifest.toml"))
    direct=["Mantle","DNNKernels","BonsaiRunner","KernelAbstractions","KernelInterface",
        "JSON3","Atomix","Adapt","Test","SHA","TOML","Statistics","LinearAlgebra"]
    selected=Set{String}();pending=copy(direct)
    while !isempty(pending)
        name=pop!(pending);name in selected && continue
        rows=manifest["deps"][name];length(rows)==1 || error("ambiguous package: $name")
        push!(selected,name);append!(pending,String.(get(only(rows),"deps",String[])))
    end
    stage=mktempdir(joinpath(root,"tmp");prefix="laptop-small-llm-")
    reduced=deepcopy(manifest)
    reduced["deps"]=Dict(name=>deepcopy(manifest["deps"][name]) for name in selected)
    # The full manifest may abbreviate weak deps as names because their UUIDs
    # occur elsewhere. The reduced closure must retain those UUIDs explicitly.
    for entries in values(reduced["deps"]), entry in entries
        weak=get(entry,"weakdeps",nothing)
        weak isa Vector || continue
        entry["weakdeps"]=Dict(name=>only(manifest["deps"][name])["uuid"] for name in weak)
    end
    project=Dict{String,Any}("deps"=>Dict(name=>only(manifest["deps"][name])["uuid"] for name in direct),
        "compat"=>Dict("julia"=>"1.12"))
    packages=Any[]
    # Copy parents before nested path dependencies such as KernelInterface.
    locals=sort!([(name,only(reduced["deps"][name])) for name in selected if haskey(only(reduced["deps"][name]),"path")];
        by=pair->length(pair[2]["path"]))
    for (name,entry) in locals
        original=abspath(joinpath(root,entry["path"]))
        relative=relpath(original,root)
        startswith(relative,"..") && error("external path dependency requires explicit mapping: $name")
        destination=joinpath(stage,relative);mkpath(dirname(destination))
        ispath(destination) || cp(original,destination;follow_symlinks=true)
        entry["path"]=relative
        push!(packages,(;name,path=relative,
            git_head=try readchomp(`git -C $original rev-parse HEAD`) catch; nothing end,
            project_sha256=bytes2hex(open(sha256,joinpath(original,"Project.toml")))))
    end
    open(io->TOML.print(io,project),joinpath(stage,"Project.toml"),"w")
    reduced["project_hash"]=Pkg.Types.workspace_resolve_hash(Pkg.Types.EnvCache(joinpath(stage,"Project.toml")))
    open(io->TOML.print(io,reduced),joinpath(stage,"Manifest.toml"),"w")
    tools=joinpath(stage,"dev/JuliaVision/tools/small_llm");mkpath(dirname(tools));cp(@__DIR__,tools)
    cp(joinpath(dirname(@__DIR__),"measure.jl"),joinpath(dirname(tools),"measure.jl"))
    model=joinpath(stage,"gen/qwen35-4b");mkpath(model)
    # Reflinks avoid duplicating multi-GB immutable weights where supported;
    # this is a data copy, never a writable hard link to a live source file.
    if include_native_model
        for name in readdir(joinpath(root,"gen/qwen35-4b"))
            name==".cache" && continue
            run(`cp --reflink=auto -a $(joinpath(root,"gen/qwen35-4b",name)) $model`)
        end
    end
    report=(;scope="pinned portable snapshot, no laptop or Bosgame live-environment changes",
        source_root=root,stage,package_count=length(selected),packages,include_native_model,
        source_manifest_sha256=bytes2hex(open(sha256,joinpath(root,"Manifest.toml"))),
        portable_manifest_sha256=bytes2hex(open(sha256,joinpath(stage,"Manifest.toml"))),
        builder_sha256=bytes2hex(open(sha256,@__FILE__)))
    open(io->JSON3.pretty(io,report),joinpath(stage,"bundle_provenance.json"),"w")
    report
end

function repair_bundle_weakdeps!(directory,source_root)
    source=TOML.parsefile(joinpath(source_root,"Manifest.toml"))
    path=joinpath(directory,"Manifest.toml");manifest=TOML.parsefile(path)
    for entries in values(manifest["deps"]),entry in entries
        weak=get(entry,"weakdeps",nothing)
        weak isa Vector || continue
        entry["weakdeps"]=Dict(name=>only(source["deps"][name])["uuid"] for name in weak)
    end
    open(io->TOML.print(io,manifest),path,"w")
    Pkg.Types.read_manifest(path)
    path
end
