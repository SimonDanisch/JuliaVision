using JSON3,SHA
alias3_digest(path)=bytes2hex(sha256(read(path)))

function freeze_alias_transfer3(;directory=joinpath(@__DIR__,"alias_transfer3_results"))
    ispath(directory)&&!isempty(readdir(directory))&&error("Preserve registration")
    preparedpath=joinpath(@__DIR__,"alias_transfer3_preparation.json");p=JSON3.read(read(preparedpath,String))
    p.complete&&!p.ready_for_generation&&length(p.jobs)==16||error("Preparation incomplete")
    alias3_digest(joinpath(@__DIR__,"alias_transfer3_protocol.jl"))==p.preparation_source_sha256||error("Preparation source drift")
    proofpath=joinpath(@__DIR__,"alias_transfer3_preparation_validation.json");proof=JSON3.read(read(proofpath,String))
    proof.complete&&proof.protocol_checks==42&&proof.snapshot_checks==4&&proof.original_reference_tasks_passed==8||error("Preparation unqualified")
    for (f,h) in pairs(proof.sources)
        alias3_digest(joinpath(@__DIR__,String(f)))==h||error("Qualification source drift")
    end
    alias3_digest(joinpath(@__DIR__,"julia_public_binding_docs.json"))==p.corpus_sha256||error("Corpus drift")
    alias3_digest(joinpath(@__DIR__,"alias_document_retrieval_validation.json"))==p.retrieval_validation_sha256||error("Retrieval proof drift")
    secureproofpath=joinpath(@__DIR__,"bigcodebench_x_secure_audit_validation.json")
    secureproof=JSON3.read(read(secureproofpath,String))
    secureproof.complete&&secureproof.tests==4||error("Test-blind runtime unqualified")
    for (f,h) in pairs(secureproof.source_sha256)
        alias3_digest(joinpath(@__DIR__,String(f)))==h||error("Test-blind qualification source drift")
    end
    parent=JSON3.read(read(joinpath(@__DIR__,"julia_stencil_transfer_results/conditions.json"),String))
    runtime=merge(Dict(String(k)=>String(v) for (k,v) in pairs(parent.source_sha256)),
        Dict(f=>alias3_digest(joinpath(@__DIR__,f)) for f in ("alias_transfer3_study.jl","run_alias_transfer3.jl")))
    snapshot=abspath(joinpath(@__DIR__,"../../../../gen/bigcodebench-x-transfer3-20261025"))
    alias3_digest(joinpath(snapshot,"manifest.json"))==p.snapshot_manifest_sha256||error("Snapshot drift")
    alias3_digest(joinpath(snapshot,"reference_validation.json"))==p.reference_validation_sha256||error("Reference drift")
    root=abspath(joinpath(@__DIR__,"../../../.."))
    packet=(;scope="Prospective third fixed-hash public BigCodeBench-X cohort, excluding prior16 tasks. Local Julia1.12/package adaptation, not official upstream/container or contamination-free/frontier/plot-aesthetics evidence.",
        policy=p.policy*" ALL16 jobs frozen before generation; one unmodified answer per arm, no retries. Independent test-blind CPU jail uses unchanged original external tests. Nonthinking output avoids reserving part of this budget for a reasoning transcript. Not a causal comparison with older thinking-budget cohorts.",
        jobs=p.jobs,decoding=p.decoding,portable_manifest_sha256=parent.portable_manifest_sha256,
        source_sha256=runtime,preparation_sha256=alias3_digest(preparedpath),preparation_validation_sha256=alias3_digest(proofpath),
        snapshot_manifest_sha256=p.snapshot_manifest_sha256,reference_validation_sha256=p.reference_validation_sha256,
        retrieval=(;p.retrieval...,corpus_sha256=p.corpus_sha256,validation_sha256=p.retrieval_validation_sha256),
        project_sha256=alias3_digest(joinpath(root,"Project.toml")),manifest_sha256=alias3_digest(joinpath(root,"Manifest.toml")),
        private_scorer_sha256=alias3_digest(joinpath(@__DIR__,"score_alias_transfer3.py")),
        secure_runtime_validation_sha256=alias3_digest(joinpath(@__DIR__,"bigcodebench_x_secure_audit_validation.json")),
        secure_runtime_sources=Dict(f=>alias3_digest(joinpath(@__DIR__,f)) for f in
            ("bigcodebench_x_secure_audit.py","bigcodebench_x_julia_adapter.py","qualify_bigcodebench_x_reference.py")))
    mkpath(directory);open(io->JSON3.pretty(io,packet),joinpath(directory,"conditions.json"),"w");packet
end
