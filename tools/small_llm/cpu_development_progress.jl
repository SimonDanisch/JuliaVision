using JSON3,SHA

"""Inventory only. Never promote infrastructure checks into model-quality gains."""
function cpu_development_progress()
    files=("alias_document_retrieval_validation.json","alias_document_ranking_audit.json",
        "rpn_example_semantics_software_validation.json","rpn_example_constraint_software_validation.json",
        "sampled_example_rpn_options_validation.json","laptop_example_constraint_gate/result.json",
        "laptop_example_identity_gate/result.json","rpn_format_followup_audit.json",
        "rpn_public_example_reference_validation.json","rpn_public_example_study_validation.json",
        "julia_stencil_transfer_reference_validation.json","julia_stencil_transfer_harness_validation.json",
        "lint_revision_protocol_validation.json","static_import_candidates_validation.json",
        "workflow_cost_summary_validation.json","julia_docs_v2_validation.json",
        "pubmedqa_reading_validation.json","pubmedqa_retrieval_validation.json",
        "safe_plot_builder_validation.json","safe_plot_julia_validation.json",
        "plot_interface_harness_validation.json","plot_render_audit_validation.json",
        "plot_nonexecution_validation.json","plot_interpretation_validation.json",
        "paired_workflow_uncertainty_validation.json","bayes_extraction_reference_validation.json",
        "bayes_extraction_study_validation.json","plot_validator_policy_validation.json",
        "plot_validator_study_validation.json","plot_validator_summary_validation.json",
        "plot_validator_deployment_validation.json","bayes_precision_audit_validation.json",
        "bayes_native_parity_software_validation.json","bayes_native_preparation_validation.json",
        "bayes_thinking_admission_validation.json","bayes_thinking_study_validation.json",
        "bayes_thinking_summary_validation.json","compact_bayes_json_validation.json",
        "compact_bayes_reference_validation.json","compact_bayes_interface_validation.json",
        "compact_bayes_study_validation.json","compact_bayes_summary_validation.json",
        "bayes_thinking_native_preparation_validation.json","finite_choice_software_validation.json",
        "finite_choice_native_gate/result.json","finite_choice_model_checks_validation.json",
        "finite_choice_model_gate/result.json","compact_bayes_demonstrations_validation.json",
        "compact_bayes_demo_workflow_validation.json","compact_bayes_demo_resume_validation.json",
        "missing_outcome_bounds_validation.json","compact_bayes_demo_native_preparation_validation.json",
        "compact_bayes_demo_native_audit/result.json")
    rows=[(;path=file,present=isfile(joinpath(@__DIR__,file)),
        sha256=isfile(joinpath(@__DIR__,file)) ? bytes2hex(sha256(read(joinpath(@__DIR__,file)))) : nothing,
        task_quality_evidence=false) for file in files]
    summarypath=joinpath(@__DIR__,"rpn_public_example_results/summary.json")
    numeric=isfile(summarypath) ? JSON3.read(read(summarypath,String)) : nothing
    numeric_progress=numeric===nothing ? (;status="registered_not_scored") :
        (;complete=numeric.complete,status_counts=Dict(s=>count(r->r.status==s,numeric.rows) for s in unique(r.status for r in numeric.rows)))
    result=(;not_complete=true,scope="Development/infrastructure inventory, not a pooled score or completion claim",receipts=rows,
        current_neural_results=(;corrected_rpn_contract="Complete: numeric correctness0/8 in each arm; grammar syntax8/8 vs1/8. Original1/8 grammar result retained separately, no regeneration.",
            alias_aware_retrieval="Third fixed-hash external8 cohort COMPLETE: direct0/8, binding-aware public docs0/8; one of16 outputs truncated. Negative evidence for this frozen workflow, not a general rejection of retrieval.",
            lint_document_revision="POSTHOC SAME8 COMPLETE: self-review0/8 vs static binding observations plus expanded public docs0/8, from identical original direct code; one truncated output. All16 retained, unchanged private scorer. No gain for this bundled workflow; not a general rejection of static tools, retrieval or bigger/trained models.",
            public_example_constraint=numeric_progress,
            symbolic_public_examples="3/4 independent CPU/software tests, zero neural calls; one public-example overfit retained",
            uniform_public_example_fallback="POSTHOC same4-task workflow3/4 vs baseline1/4, two repairs/no regressions; not independent transfer or model-only intelligence",
            julia_source_transfer="Prospective NEW8 related-family tasks: direct6/8, frozen public-example fallback6/8; no gain. All16 physical GPU outcomes agree.",
            boundary_contract_development="POSTHOC same8 explicit neighbor-mode contract + constrained symbolic search8/8, two repairs/no regressions",
            fresh_boundary_transfer="Prospective next8: direct7/8, example-only8/8, boundary-aware8/8; one repair, no incremental boundary accuracy gain. All24 physical GPU outcomes agree; conservative boundary rule also replaced one correct program unnecessarily."),
        biomedical_reading_pilot="COMPLETE48 outputs: corresponding abstract13/16 vs question-only7/16 and mismatched abstract7/16.7 repairs/1 regression, all-no controls, only maybe item missed, best constant8/16. Source-reading workflow effect ONLY, not a general retriever, clinical advice or frontier evidence.",
        biomedical_automatic_retrieval="COMPLETE64 outputs on NEXT16 papers: automatic body-only BM2513/16, oracle13/16, question-only/mismatch3/16 each.10 repairs/0 regressions, source15/16, repeated-prompt15groups all agree. Best constant11/16; BOTH maybe cases missed. Small retrieval/reading workflow result, NOT clinical reliability or general medical consensus.",
        makie_interface="COMPLETE: restricted Julia2/8 vs JSON3/8 specifications,1 repair/no regressions; not convincing generalized gain. ALL8 admitted plans rendered unclipped, including3 specification failures;8 unsupported/truncated outputs retained. No aesthetic preference evaluation.",
        makie_validator_followup="COMPLETE POSTHOC SAME8: original JSON3/8, conditional self-review3/8, conditional parser-feedback3/8; no repairs/regressions. PUBLIC trigger on3 originals,6 new answers all failed admission. Five admitted originals per branch, including semantic failures, unchanged. No gain for this workflow; not a general rejection of validators.",
        bayesian_extraction="COMPLETE16 outputs: direct numeric0/8, model-network plus shared solver0/8. Direct8 admitted; only1/8 networks admitted, none faithful. BOTH arms still0 at1% absolute tolerance. Independent native FP32/GREEDY CPU diagnostic also0/8 at primary tolerance,1/8 at1%; exact same chat IDs and all8 completed. Different precision AND sampler: NOT isolated Q4 effect or capability ceiling. Sole admitted generated network matches physical GPU/software,13 harness checks plus replay; numerical parity is not extraction quality.",
        bayesian_guarded_thinking="COMPLETE POSTHOC SAME8/16outputs: direct2/8 vs original0/8, network0/8 vs0/8. Four direct answers admitted, zero networks admitted;8 outputs truncated and3 completed outputs rejected for closure count. All16 forced one thinking closure. Different thinking/template/controller and2048 total allowance, not equal compute or isolated weight/mode effect; only2 repairs, descriptive paired p0.5. No admitted networks, so no physical parity job justified. Not frontier/general Bayesian reasoning or clinical competence.",
        bayesian_compact_interface="COMPLETE POSTHOC SAME8: compact CPT extraction0/8, zero admitted vs original full-schema0/8. Six completed outputs malformed/incomplete or wrong field shapes; two truncated. Host binary-complement packing/shared solver passed86 reference checks, but no generated network reached the solver. No model-network correction or task retry; shorter representation alone did not improve this workflow. Public-example follow-up is separate below; decode guidance remains untested.",
        bayesian_compact_demonstrations="POSTHOC SAME8 plus two public unrelated network-format examples: seven completed, one process-loss attempt not regenerated. Correct faithful networks2/7 completed vs bare compact0/7 on those tasks;4 admitted including2 wrong. Registered8 lower/upper success bounds2/8..3/8, not confidence intervals; lost outcome is unknown, not a semantic model failure. Two repairs/no observed regressions, descriptive discordance p0.5. All seven generation workers normally retired; original lost context has no normal-waitidle claim, fresh recovery gate86 passed. Changed public examples/context/compute, not independent transfer, model weights or frontier intelligence. ALL4 admitted networks undergo separate physical parity, including wrong ones.",
        finite_choice_device_constraint="Opt-in canonical-token finite-support prototype:39 host +47 software-Vulkan and39 host +47 physical checks PASS. Full4B one-prompt/eight-step gate PASS: disabled control matches token history/seen state/final raw-logit bits; enabled output follows one supplied API-name path/EOS; one recorded submission per decode step, no host token feedback. Finalwaitidle/cache save and retirement normal. Tool return-formatting hit a separate world-age error; durable result verified without regeneration. No task-accuracy gain, arbitrary JSON/Julia grammar, live scope extractor or semantic-choice guarantee.",
        unfinished=["Generalization beyond authored stencil-family tasks remains unestablished. Preserve prospective6/8 no-gain, posthoc8/8, and NEW8 direct7/8/fallback8/8 as separate cohorts",
            "Same-cohort lint/docs revisions completed0/8 in both arms; original direct/docs0/8 scores retained separately. Broader coding improvement remains unestablished",
            "Physical-device numerical grading of completed corrected-contract outputs remains optional and not yet performed",
            "Expanded17-module public corpus and hardened static diagnostics are infrastructure; their model-quality effect is being evaluated separately from the original7-module retrieval study",
            "Broader Makie construction/semantic fidelity and aesthetic preference remain unestablished beyond the authored8-task pilot; conditional public-validator follow-up completed without gain",
            "Original Bayesian nonthinking0/8 both arms, guarded-thinking2/8 direct and0/8 network, bare compact0/8; public examples2/7 completed plus1 unknown process-loss outcome. Independent transfer, longer natural thinking and learned latent bridges remain unestablished",
            "Finite-choice constraints passed host/software/physical fixtures and one full-model identity/support gate; task-quality effects and generalization remain untested",
            "Frontier-level capability, medical/legal/tax assistance, million-token performance and scalable latent knowledge integration remain unestablished"],
        laptop_inference="Bounded per-output inference windows with explicit final waitidle. Do NOT infer device idle from scoring status.",
        bosgame="No Radeon setup/reset/recovery attempted; Laptop fallback is independent")
    open(io->JSON3.pretty(io,result),joinpath(@__DIR__,"cpu_development_progress.json"),"w");result
end
