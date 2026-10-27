include("pubmedqa_retrieval_protocol.jl")
using JSON3,SHA
function prepare_pubmedqa_retrieval(;token_count,snapshot="gen/pubmedqa-retrieval-20261025",
        destination=joinpath(@__DIR__,"pubmedqa_retrieval_preparation.json"))
    isfile(destination)&&error("Preserve preparation")
    digest(path)=bytes2hex(sha256(read(path)))
    manifestpath=joinpath(snapshot,"manifest.json");manifest=JSON3.read(read(manifestpath,String))
    taskpath=joinpath(snapshot,"public/tasks.json");corpuspath=joinpath(snapshot,"public/corpus.json")
    manifest.complete&&manifest.count==16&&manifest.corpus_documents==1000||error("Snapshot coverage")
    manifest.public_sha256==digest(taskpath)&&manifest.corpus_sha256==digest(corpuspath)||error("Public source drift")
    manifest.snapshotter_sha256==digest(joinpath(@__DIR__,"snapshot_pubmedqa_retrieval.py"))&&
        manifest.parent_snapshotter_sha256==digest(joinpath(@__DIR__,"snapshot_pubmedqa_pilot.py"))||error("Snapshot source drift")
    tasks=JSON3.read(read(taskpath,String));corpus=JSON3.read(read(corpuspath,String))
    length(tasks)==16&&length(corpus)==1000||error("Entire cohort and corpus required")
    oldpath=joinpath(@__DIR__,"pubmedqa_reading_results/conditions.json");old=JSON3.read(read(oldpath,String))
    isempty(intersect(Set(String(t.pmid) for t in tasks),Set(String(j.task) for j in old.jobs)))||error("Prior question overlap")
    packets=PubMedQARetrievalProtocol.jobs(tasks,corpus;characters=8000)
    jobs=map(packets) do job
        tokens=token_count(job.messages)
        tokens isa Integer&&!(tokens isa Bool)&&tokens>0||error("Invalid token count")
        tokens+128+32<=8192||error("Capacity failure: do not filter cases")
        (;job...,input_tokens=tokens)
    end
    sources=("prepare_pubmedqa_retrieval.jl","pubmedqa_retrieval_protocol.jl","pubmedqa_reading_protocol.jl",
        "public_document_retrieval.jl","snapshot_pubmedqa_retrieval.py","snapshot_pubmedqa_pilot.py")
    result=(;complete=true,ready_for_generation=false,jobs,decoding=old.decoding,
        scope="Next disjoint fixed-hash16 public PubMedQA questions, not the official split or contamination-free clinical evaluation",
        policy="ALL64 four-arm answers frozen before generation; one seed/max128 nonthinking tokens, no retries/hidden feedback/task selection. Question-only, oracle corresponding context, next-question mismatch, AUTOMATIC body-only BM25 top1 over ALL1000 source abstracts. No article titles, conclusions, prior predictions or gold labels in index. Same8000-character cap/one-string paragraph format for every evidence arm. Four-arm Latin order; source selection never asks the model to formulate a query. Strict one-word scoring unchanged; maybe is not confidence.",
        snapshot_manifest_sha256=digest(manifestpath),public_sha256=digest(taskpath),corpus_sha256=digest(corpuspath),
        prior_conditions_sha256=digest(oldpath),sources=Dict(file=>digest(joinpath(@__DIR__,file)) for file in sources),
        remaining="Qualify actual-corpus shared scoring and immutable four-arm runner/scorer, then register/deploy before answers")
    open(io->JSON3.pretty(io,result),destination,"w");result
end
