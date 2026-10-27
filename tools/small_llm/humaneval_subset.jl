# Read-only adapter. NEVER include/eval the external runner or task files in
# Main; trusted gold code and model candidates execute only in our sandbox.
using JSON3, SHA
isdefined(@__MODULE__,:JuliaTaskEval) || include("julia_eval.jl")
const HUMANEVAL_REVISION="52b58aa40d63b1a6c08ac910b798229b36f5c24f"

function read_humaneval_task(root,file)
    occursin(r"^[0-9]{3}_[A-Za-z0-9_]+\.jl$",file) || throw(ArgumentError("task filename"))
    source=read(joinpath(root,"src/tasks",file),String)
    doc=match(r"\"\"\"(.*?)\"\"\""s,source)
    doc===nothing && error("missing public docstring: $file")
    testpath=joinpath(root,"test/basic",replace(file,".jl"=>"_test.jl"))
    tests=Meta.parseall(read(testpath,String))
    statements=filter(x->!(x isa LineNumberNode),tests.args)
    length(statements)==1 || error("unexpected test file structure")
    item=only(statements)
    item isa Expr && item.head==:macrocall && item.args[1]==Symbol("@testitem") || error("unexpected test macro")
    body=last(item.args)
    body isa Expr && body.head==:block || error("unexpected test body")
    expected=Meta.parse("include(joinpath(ENV[\"GENERATION_DIR\"], $(repr(file))))")
    count(x->x==expected,body.args)==1 || error("expected exactly one candidate include")
    content=filter(x->!(x isa LineNumberNode)&&x!=expected,body.args)
    imports=filter(x->x isa Expr&&x.head in (:using,:import),content)
    checks=filter(x->!(x isa Expr&&x.head in (:using,:import)),content)
    ast_source(xs)=join((sprint(Base.show_unquoted,x) for x in xs),"\n")
    (;id=splitext(file)[1],category="base",filename=file,
        prompt="Implement the documented Julia function below. Return its COMPLETE definition, not just a function body. Include required standard-library imports. Code only.\n\n"*String(doc.captures[1]),
        setup=ast_source(imports),tests=ast_source(checks),reference=source,
        original_source_sha256=bytes2hex(sha256(source)),original_tests_sha256=bytes2hex(sha256(read(testpath))))
end

function load_frozen_humaneval_tasks(;root="tmp/humaneval-julia")
    readchomp(`git -C $root rev-parse HEAD`)==HUMANEVAL_REVISION || error("external revision mismatch")
    path=joinpath(@__DIR__,"humaneval_subset_results/registration.json")
    packet=JSON3.read(read(path,String));tasks=Any[]
    for row in packet.rows
        task=read_humaneval_task(root,String(row.filename))
        task.original_source_sha256==row.source_sha256 || error("external source drift")
        task.original_tests_sha256==row.original_tests_sha256 || error("external tests drift")
        bytes2hex(sha256(task.tests))==row.adapter_tests_sha256 || error("test adapter drift")
        task.prompt==row.prompt || error("public prompt drift")
        push!(tasks,task)
    end
    tasks
end

function requalify_humaneval_subset(;root="tmp/humaneval-julia",version="v2")
    occursin(r"^v[0-9]+$",version) || throw(ArgumentError("version"))
    directory=joinpath(@__DIR__,"humaneval_subset_results")
    output=joinpath(directory,"qualification_"*version*".json")
    isfile(output) && error("qualification attempt already recorded")
    rows=Any[]
    for task in load_frozen_humaneval_tasks(;root)
        score=JuliaTaskEval.evaluate(task,task.reference;output=joinpath(directory,task.id*"_gold_"*version*"_score.json"))
        push!(rows,(;id=task.id,passed=score.passed,elapsed_seconds=score.elapsed_seconds))
    end
    report=(;scope="same frozen subset, requalification after environment repair; original negative qualification retained",
        selected=length(rows),qualified=count(r->r.passed,rows),rows)
    open(io->JSON3.pretty(io,report),output,"w")
    report
end

function freeze_humaneval_subset(;root="tmp/humaneval-julia",count=12)
    readchomp(`git -C $root rev-parse HEAD`)==HUMANEVAL_REVISION || error("external revision mismatch")
    1<=count<=164 || throw(ArgumentError("subset size"))
    directory=joinpath(@__DIR__,"humaneval_subset_results");mkpath(directory)
    registration=joinpath(directory,"registration.json")
    isfile(registration) && error("subset already frozen; do not select again")
    files=filter(f->endswith(f,".jl"),readdir(joinpath(root,"src/tasks")))
    sort!(files;by=f->bytes2hex(sha256("small-llm-julia-external-v1:"*f)))
    selected=files[1:count]
    tasks=[read_humaneval_task(root,file) for file in selected]
    packet=(;repository="https://github.com/01-ai/HumanEval.jl",revision=HUMANEVAL_REVISION,
        license="MIT; Copyright(c)2024 01.AI; original LICENSE retained in pinned checkout",
        selection="first$count filenames sorted by SHA256(small-llm-julia-external-v1:<filename>), frozen before model generation",
        limitations=["public benchmark, training contamination possible","basic upstream tests only, NOT evalplus or full164-task benchmark"],
        rows=[(;id=t.id,filename=t.filename,prompt=t.prompt,source_sha256=t.original_source_sha256,
            original_tests_sha256=t.original_tests_sha256,adapter_tests_sha256=bytes2hex(sha256(t.tests))) for t in tasks])
    open(io->JSON3.pretty(io,packet),registration,"w")
    # Persist the selection BEFORE checking gold implementations. Unqualified
    # cases remain explicitly unqualified; never silently replace them.
    rows=Any[]
    for task in tasks
        score=JuliaTaskEval.evaluate(task,task.reference;output=joinpath(directory,task.id*"_gold_score.json"))
        push!(rows,(;id=task.id,passed=score.passed,elapsed_seconds=score.elapsed_seconds))
        println(task.id," gold ",score.passed ? "PASS" : "UNQUALIFIED");flush(stdout)
    end
    report=(;scope="sandbox/adapter qualification only; no LLM quality result",selected=length(tasks),qualified=Base.count(r->r.passed,rows),rows)
    open(io->JSON3.pretty(io,report),joinpath(directory,"qualification.json"),"w")
    report
end
