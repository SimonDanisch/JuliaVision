include("pubmedqa_reading_protocol.jl")
using JSON3,SHA
function prepare_pubmedqa_reading(;token_count,snapshot="gen/pubmedqa-reading-pilot-20261025",
        destination=joinpath(@__DIR__,"pubmedqa_reading_preparation.json"))
    isfile(destination)&&error("Preserve preparation")
    digest(path)=bytes2hex(sha256(read(path)))
    manifestpath=joinpath(snapshot,"manifest.json");manifest=JSON3.read(read(manifestpath,String))
    publicpath=joinpath(snapshot,"public/tasks.json")
    manifest.complete&&manifest.count==16&&manifest.public_sha256==digest(publicpath)||error("Public snapshot provenance")
    manifest.snapshotter_sha256==digest(joinpath(@__DIR__,"snapshot_pubmedqa_pilot.py"))||error("Snapshotter drift")
    tasks=JSON3.read(read(publicpath,String));rows=PubMedQAReadingProtocol.jobs(tasks)
    jobs=map(rows) do row
        count=token_count(row.messages)
        count isa Integer&&!(count isa Bool)&&count>0||error("Invalid tokenizer result")
        count+128+32<=8192||error("Capacity failure; no score/length filtering or partial registration")
        (;row...,input_tokens=count)
    end
    result=(;complete=true,ready_for_generation=false,jobs,
        scope="Fixed-hash16 public expert-labeled PubMedQA items; corresponding context is oracle evidence. Not official split/leaderboard, contamination-free or clinical competence.",
        policy="ALL48 conditions frozen before generation. Same seed/sampler/max128 nonthinking tokens, counterbalanced three-arm order; one raw answer per arm/task, no retries/hidden feedback. Strict trimmed case-normalized yes/no/maybe only; punctuation/explanations/truncation/errors invalid. Corresponding vs next-item context vs question-only diagnostic. maybe is a study label, not calibrated uncertainty. Private gold and conclusions NEVER deployed.",
        decoding=(;max_tokens=128,thinking=false,seed=20261025,topk=20,temperature=0.6f0,top_p=0.95f0,presence=0f0,output_chunk=64),
        public_sha256=digest(publicpath),snapshot_manifest_sha256=digest(manifestpath),
        protocol_sha256=digest(joinpath(@__DIR__,"pubmedqa_reading_protocol.jl")),
        preparer_sha256=digest(@__FILE__),remaining="Qualify runner/scorer, then freeze source/model/runtime/conditions BEFORE ANY pilot answers")
    open(io->JSON3.pretty(io,result),destination,"w");result
end
