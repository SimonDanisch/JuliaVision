# Small language models with external reasoning and memory

Research and local integration assessment, 24 October 2026.

Start with **Qwen3.5-4B**, retain Qwen3.5-2B as the throughput fallback, and compare external evidence at three interfaces: text context, token probabilities, and intermediate model representations. Keep one million tokens as an explicit ingestion and recall test. A separate experiment should use a smaller active context with a million-token external store, because fitting the full context in memory does not establish fast decoding or reliable recall.

The research supports separating memory from computation. It does not establish that removing factual knowledge from a pretrained model preserves its reasoning ability. Existing weights jointly encode language, facts, representations and procedures; a useful first experiment adds interfaces to a capable small backbone rather than trying to subtract its knowledge.

## Local hardware and repository state

The local host is an AMD Ryzen AI Max+ 395 with Radeon 8060S and approximately 122 GiB of OS-visible RAM. This is shared system and GPU memory. At inspection, about 55 GiB was available and swap was already in use; the full installed memory was not available for a new session.

The following development branches were fetched on 24 October:

| Component | Commit | Working location |
| --- | --- | --- |
| JuliaVision | `823ba9da4150537ce4a4a8a045d75ce64ca42ab4` | `dev/JuliaVision`, fast-forwarded on `sd/mantle-dev` |
| Mantle | `be46b479694c1fd6fc8234160db9eeabad98110b` | `dev/Mantle`, fast-forwarded on `sd/lava-refactor` |
| Lava | `4efa0527093ead226866869361c5c30037f311f7` | `dev/Lava`, fast-forwarded on `sd/lava-refactor` |
| KernelAbstractions | `79ac658` | `dev/KernelAbstractions`, fast-forwarded on `sd/lava-refactor` |
| Metal dependency | `f0280e98` | `dev/Metal`, `sd/lava-refactor` |

The unrelated spatial work is preserved in local branch `tmp/park-spatial-2026-10-24`, commit `7e601c5` (38 files, including previously untracked examples and tests). The active Mantle checkout is clean. All subsequent Julia work uses the workspace root environment, `/home/sim/Programmieren/VulkanDev`. The earlier `tmp/small-llm-env` is no longer used. `Pkg.update()` upgraded the root dependency graph to GPUArrays 12.1.1, KernelAbstractions 0.10.1 and KernelInterface 0.4.1; AMDGPU uses `mantle/ka-0.10` and Metal uses `sd/lava-refactor`. The root project no longer includes WaterLily, WaterLilyMeshBodies, BiotSavartBCs or GPUSelect. Their checkouts remain on disk. Adapt's local constructor fix was retained while merging current upstream changes into its existing branch (merge commit `2651d56`).

Existing local runtime results provide useful baselines, not measurements of the candidate Qwen models:

| Runner | Existing recorded result on Radeon 8060S | Implication |
| --- | --- | --- |
| Bonsai 2 27B PTQ1 | 29.3 decode tokens/s, 10 October | Already supported; below the requested 50–100 tokens/s |
| Horizon 32B Q4_K_M | 108.7 ms/token, approximately 9.2 tokens/s, 13 October | Too slow for the exploration backbone |

Sources: [BonsaiRunner README](../../../BonsaiRunner/README.md), [HorizonRunner README](../../../HorizonRunner/README.md). These are short-context measurements. The new benchmark script records its own timing samples and clock readings separately.

The fresh root-environment run completed on 24 October: five repetitions of 128 decode steps after the same prompt produced identical token sequences and finite logits. Median throughput was **9.22 tokens/s**, ranging from **8.99 to 9.33** across accepted samples. A concurrent GPU recording workload was observed on the machine during the run. Treat this as a successful integration check under contention, not a quiet-machine performance baseline or evidence of a regression against the historical 29.3 tokens/s. Raw samples are in [bonsai_baseline.json](../../../tools/small_llm/bonsai_baseline.json).

## Research closest to the proposed architecture

| Interface | Research and result | What it establishes and what remains |
| --- | --- | --- |
| Retrieved evidence changes next-token probabilities | [kNN-LM, 2019/2020](https://arxiv.org/abs/1911.00172) interpolates the neural token distribution with a distribution from neighbors in hidden-state space, improving perplexity without additional model training. | Direct precedent for influencing the probability space without a tool transcript. The datastore needs model-specific representations; improved perplexity is not a demonstration of general reasoning. |
| Retrieval participates within a forward pass | [RETRO, 2021](https://arxiv.org/abs/2112.04426) combines chunk retrieval with cross-attention. Its 7.5B model was competitive with much larger models on language-modeling benchmarks. | Evidence for a smaller computation backbone plus large external memory. Requires a trained retrieval interface; the result does not imply modern frontier reasoning at 7.5B. |
| Internal activations become retrievable memory | [Memorizing Transformers, 2022](https://arxiv.org/abs/2203.08913) retrieves stored attention keys and values and improves language modeling as memory grows. | Close to memory inside the model rather than repeated textual lookup. Arbitrary database embeddings cannot simply be inserted as native layer-specific KV entries. |
| Formal syntax constrains each sampling step | [PICARD, 2021](https://arxiv.org/abs/2109.05093) rejects token extensions that fail incremental parsing, with strong text-to-SQL results. | Cheap, testable first intervention. Valid syntax does not imply the requested program or query is correct. |
| IDE information constrains each sampling step | [Monitor-Guided Decoding, 2023](https://arxiv.org/abs/2306.10763) uses static analysis and language-server results to restrict identifiers and other code choices. Smaller augmented models can outperform larger baselines on its evaluated measures. | Very close to the language-server proposal. Partial programs, subword tokens and incomplete completion lists require careful handling. |
| Logical requirements guide search | [NeuroLogic Decoding, 2021](https://aclanthology.org/2021.naacl-main.339/) enforces combinations of lexical constraints; [NeuroLogic A*esque](https://arxiv.org/abs/2112.08726) adds lookahead. | Shows why locally masking tokens alone may be insufficient. Lexical predicate constraints are not a general Prolog theorem prover. |
| Executable symbolic programs replace free-form arithmetic | [NeuroProlog, March 2026, revised September](https://arxiv.org/abs/2603.02504) trains translation into executable Prolog and evaluates execution-guided repair. | Particularly relevant to small models. Its analysis still finds semantic failures in small models: a solver cannot correct a wrongly translated problem by itself. |
| Symbolic transition models guide generation and search | [Neurosymbolic Embodied Agents, August 2026](https://arxiv.org/abs/2608.16794) combines PDDL constraints with Monte Carlo tree search for open 4B–27B models. | Strong domain-specific evidence that constraints and search can complement small models. Executability is conditional on correct grounding and transition rules. |
| Reasoning is integrated into diffusion inference | [SearchDiff, February 2026](https://arxiv.org/abs/2602.02727) searches proposal sets during denoising to improve constraint satisfaction. | A broader architectural option, but the current local runners are autoregressive; this would need a separate inference path. |
| Graphs supply structured retrieval | [GraphRAG](https://arxiv.org/abs/2404.16130) uses graph communities and summaries for global questions; [GNN-RAG](https://aclanthology.org/2025.findings-acl.856/) uses a GNN to retrieve graph evidence efficiently. | Useful baselines for database access. Their cited pipelines provide textual evidence to the LLM and are distinct from latent graph fusion. |
| Static memory has its own capacity budget | [Memory Layers at Scale, 2024/2025](https://arxiv.org/abs/2412.09764) scales trainable sparse lookup memory with strong factual-task gains; [Engram, January 2026, revised July](https://arxiv.org/abs/2601.07372) adds hashed n-gram memory with contextual gating. | Direct support for separating lookup capacity from neural computation. Both involve trained memory; they do not turn an arbitrary database into a plug-in for an unchanged checkpoint. |
| Memory can be updated independently | [EngramEdit, 7 October 2026](https://arxiv.org/abs/2610.10533) edits conditional memory while preserving the backbone and evaluates paraphrases, unrelated knowledge and multi-hop use. | Especially close to editable knowledge modules. It depends on a model equipped with conditional memory and is a recent preprint. |
| Neural long-term memory learns during inference | [Titans](https://arxiv.org/abs/2501.00663) combines attention with a learned memory updated at test time. | Relevant to persistent memory. A different model architecture and training setup, not a cache configuration flag for Qwen. |

The strongest precedents are therefore more specific than ordinary retrieval-augmented prompting: kNN-LM for probabilities, monitor-guided decoding for code, RETRO and GreaseLM for internal fusion, and Engram for separate lookup capacity.

## Bringing logic and databases into model space

This reverse direction is feasible in several different senses. They should be evaluated separately.

**Encode structured state as latent memory.** A small encoder converts a retrieved graph neighborhood, proof state, or probabilistic factor graph into vectors. A trained projection makes them available through cross-attention at selected backbone layers. [QA-GNN](https://arxiv.org/abs/2104.06378) jointly reasons over language and graph context; [GreaseLM](https://arxiv.org/abs/2201.08860) exchanges information between graph and language representations across multiple layers. These establish the mechanism on their QA tasks, rather than universal symbolic reasoning.

**Make logic itself differentiable.** [Neural Theorem Provers](https://arxiv.org/abs/1705.11040) replace exact symbol unification with similarity between learned representations. [Logic Tensor Networks](https://arxiv.org/abs/2012.13635) interpret logical formulas with differentiable many-valued semantics; [sLTN, August 2026](https://arxiv.org/abs/2608.11136) makes structural tensor axes explicit. These allow gradients to train representations from constraints. Similarity and fuzzy satisfaction have different semantics from exact logical truth and from Bayesian probability.

**Learn probabilistic predicates and infer with them.** [DeepProbLog](https://arxiv.org/abs/1805.10872) integrates neural predicates with probabilistic logic. For our interface, posterior distributions, variable identities, observations and factor dependencies can be encoded into memory vectors. A list of marginal probabilities alone may lose dependencies that matter for a later question. Preserve the factor graph or exact inference state alongside its neural representation.

**Distill reusable procedures into small modules.** [K-Adapter](https://arxiv.org/abs/2002.01808) adds knowledge-specific adapters to a frozen language backbone; [AdapterFusion](https://arxiv.org/abs/2005.00247) composes trained adapters. These support the idea of language or domain add-ons, though the original studies use earlier encoder models. An adapter still stores learned information in parameters. Frequently changing facts are better candidates for external memory, while stable syntax, query construction and reasoning procedures are candidates for adapters.

For a first latent interface, expose a bounded set of memory slots with explicit source identity, variable bindings, validity time, relation type, confidence and proof status. Train a gated cross-attention adapter on solver-generated examples while freezing the backbone. A conceptual update is:

`h_next = transformer_block(h) + gate(h, memory) * cross_attention(h, encode(structured_state))`

This is a proposed experiment, not an implemented layer. Its projection, normalization and gate must be trained: an arbitrary graph embedding is not automatically meaningful to Qwen's residual stream. Train and test with renamed entities, unseen graph structures, deeper proofs and changed facts to detect memorized shortcuts. Exact verification should consume the structured state, since compressing it into vectors does not preserve formal guarantees by itself.

## Model choice and the throughput target

The initial shortlist uses ungated Apache-2.0 checkpoints from the official Qwen organization. Exact revisions are pinned in [models.toml](../../../tools/small_llm/models.toml).

| Model | Role | Context and integration |
| --- | --- | --- |
| [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | First dense backbone to port | 262,144 native tokens; the model card documents YaRN extension to 1,010,000. 32 layers, including 24 recurrent layers and 8 full-attention layers. |
| [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | Faster fallback for exploration | 262,144 native positions in the pinned configuration; smaller full-attention cache. Verify long-context quality independently. |
| [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | Higher-capacity comparison after the dense model | 35B total, about 3B active parameters; 262K native and documented extension to 1,010,000. Requires routed expert loading and efficient MoE inference. |

The 4B name does not mean the entire multimodal checkpoint has exactly four billion parameters: its Hub metadata lists about 4.66B including the vision component. Start with text inference. Published capability claims and active parameter counts do not establish performance on this GPU or this runtime.

**The 4B runner meets the 50–100 tokens/s target at short context; 2B exceeds it.** These measured results are detailed below. For initial sizing, the local Horizon documentation reports roughly 220 GB/s of weight streaming. Under that bandwidth assumption, a hypothetical 2.5 GB of streamed weights costs about 11.4 ms/token before attention, state updates and dispatch overhead. This sizing estimate alone does not predict actual model throughput.

DNNKernels supplies Q4_K and Q6_K matrix operations, Gated DeltaNet updates, causal convolution, normalization, rotary embeddings and attention. BonsaiRunner demonstrates recorded Mantle decode, Q8 KV paging and chunked prefill, but its implementation hardcodes Bonsai-specific dimensions and Hadamard/PTQ1 weights. The experimental [Qwen text runner](../../../tools/small_llm/qwen35.jl) now loads both native Qwen3.5 checkpoints and handles their contiguous recurrent-head repetition separately from Bonsai's converted layout.

The runner supports FP32 and Q4_K projections with a Q6_K tied output head, sequential and chunked prompt ingestion, FP32 KV storage and native RoPE. YaRN and Q8 KV remain integration work. Native tokenizer/chat inputs are checked against independent local Transformers/Rust-tokenizer references.

### Measured Qwen decode, 24 October

| Model | Prompt tokens | Decode tokens/s, median | Observed accepted range | Work per sample |
| --- | ---: | ---: | ---: | --- |
| Qwen3.5-2B Q4_K/Q6_K | 26 | 124.41 | 123.88–125.18 | 64 decode steps, 3 samples |
| Qwen3.5-4B Q4_K/Q6_K | 26 | 63.25 | 63.17–63.34 | 128 decode steps, 5 samples, 4 accepted by clock filter |
| Qwen3.5-4B Q4_K/Q6_K | 4096 | 51.06 | 51.05–51.06 | 128 decode steps, 3 samples |

The 26-token tests use 4096-token cache capacity; the occupied 4096-token test uses capacity 8192. Capacity is not occupied context. Raw results: [2B](../../../tools/small_llm/qwen2b_q4_resident_benchmark.json), [4B](../../../tools/small_llm/qwen4b_q4_resident_benchmark.json), [4B with 4K prompt](../../../tools/small_llm/qwen4b_q4_4k_benchmark.json). Compilation, model loading, graph recording and prompt ingestion are excluded from decode timing.

The 4K prompt repeats and truncates the 26-token chat input to establish occupied attention cache. This is a runtime measurement, not a long-context understanding evaluation. Sequential ingestion took a median 69.83 seconds, about 58.7 prompt tokens/s. Implementing chunked/batched prefill is a prerequisite to useful large-context ingestion.

The first [chunked prefill implementation](../../../tools/small_llm/prefill_qwen35.jl) now batches projections/MLPs through existing eight-column quantized kernels and evaluates the vocabulary head only for the chunk's final token. It shares the validated decode recurrence and attention kernels and retains KV/recurrent state between chunks. A 32-token chunk reduced median 4B ingestion for the same 4096-token prompt to **12.35 seconds** (about **331.6 prompt tokens/s**, **5.65x** faster), while decode remained **50.81 tokens/s**. All three samples passed determinism, finite-logit and one-submission-per-decode checks; [raw results](../../../tools/small_llm/qwen4b_q4_4k_chunked_benchmark.json). This is not yet the parallel DeltaNet recurrence or a cooperative-matrix prefill implementation.

The [prefill parity suite](../../../tools/small_llm/test_prefill_qwen35.jl) passed **237 checks** on 2B Q4: last-token logits, every layer, convolution/recurrent state, KV cache, continuation tokens, nonempty prefixes and chunk/tail boundaries. The [parallel recurrence CPU oracle](../../../tools/small_llm/delta_chunk_reference.jl) separately passed **74 checks** against sequential state updates, including zero decay, no writing, retained state, extreme underflow and chunk composition. Its lower-triangular solve is groundwork for a later GPU implementation. The [original Gated DeltaNet paper](https://arxiv.org/abs/2412.06464) derives a hardware-efficient chunkwise algorithm; the [Transformers issue tracker](https://github.com/huggingface/transformers/issues/46190) also records reports of chunk/decode divergence, reinforcing the need for real-checkpoint state parity rather than random tensor checks alone.

The [native tokenizer wrapper](../../../tools/small_llm/qwen_text.jl) reuses BonsaiRunner's byte BPE with native checkpoint vocabulary/merges and NFC normalization. **20 CPU checks** cover saved Transformers prompt IDs, chat-template parity, round trips and independent Rust-tokenizer fixtures for multilingual text, combining characters, whitespace, control bytes and added tokens. The chat renderer currently supports plain text roles; full tool/multimodal templates remain out of scope for this prototype.

Every sample produced identical token IDs and finite logits. The measured Vulkan submission count was exactly one per decode step, including 128 submissions for each 128-step 4B sample. Each decode graph includes all layers, the output head, argmax, token feedback and position advancement; it is recorded with `budget=Inf`. Generation does not read the chosen token to the CPU between steps. The token history is read after the batch. Independent feedback checks match explicit token feeding for both model sizes; the 2B FP32 recheck passes all 56 reference checks, including all 24 layers.

The earlier roughly 3.2 tokens/s 4B observations had concurrent GPU work and the earlier host-readback/default-budget path. They cannot isolate the effect of removing readback or changing submission batching. Use the coordinated recorded-graph measurements above as the new short-context baseline.

The [evidence interface](../../../tools/small_llm/knowledge_bias.jl) declares log-likelihood additions or hard token constraints into the same model graph through `session(...; logit_hook=...)`, before argmax and GPU feedback. Evidence updates are queued stores in the next decode submission. Seven small arithmetic checks verify the finite Bayesian update and constraint validation; four model checks verify that changing the mask changes subsequent token feedback while retaining one submission per step. The separate `Decoder` graph is an arithmetic test fixture, not the generation path. These checks establish the interface, not improved reasoning or a trained latent knowledge representation.

The runtime [portable evidence module](../../../tools/small_llm/portable_evidence.jl) implements finite-factor inference, Horn-rule closure, proof provenance and evidence-to-logit operations once as KernelInterface kernels recorded through Mantle. **The same kernel functions run on CPU through Lavapipe and on the Radeon GPU.** Host code only packs symbolic variable/rule names into indexed arrays. The independent dictionary/set solver now lives under [reference/](../../../tools/small_llm/reference/structured_evidence.jl) and is used only by tests; runtime logic experiments no longer call it.

Shared-kernel validation passed **78 checks per device** (30 factor-inference, 42 proof/status-mask, 6 Bayesian-to-logit), plus **33 CPU/GPU output comparisons**. Tests cover impossible/zero-mass evidence, undefined conditionals, extreme factor scales, updated observations, ungrounded cycles, empty-premise facts, unknown/refuted distinctions, contradictions, proof provenance, and combined inference/readout graphs with exactly one submission. Factor log scales are centered before Float32 conversion, since common multiplicative factor scales cancel from normalized results and evidence probabilities. Results remain subject to Float32 rounding; exact enumeration refers to the algorithm, not exact numerical arithmetic. Enumeration remains explicitly budgeted and Horn closure uses a bounded synchronous iteration over finite ground atoms.

Both `declare_horn!` and `declare_bayes!` can be embedded directly in a model's logit hook. The Horn result drives a device-side truth/status mask; computed conditional Bayesian likelihoods can update logits through `declare_bayesian_logits!` without an intervening host readback. Unknown/conflicting world-truth queries do not become hard yes/no masks; impossible evidence and undefined likelihoods leave the model's logits unchanged. Candidate restriction is explicit, and Bayesian interpretation still requires a meaningful hypothesis prior.

The [updated logic pilot](../../../tools/small_llm/qwen2b_shared_kernel_logic_pilot.json) ran the shared rule kernels, status readout and neural selection inside the same model graph. Its four formal status cases all matched the exact kernel result. This is an integration ceiling with known formal inputs, not evidence of general reasoning improvement; the text-only arm solved one of four cases. Larger held-out task evaluation and learned latent interfaces remain future experiments. Load [load_evidence_checks.jl](../../../tools/small_llm/load_evidence_checks.jl) in the root environment to run the same suite with `Mantle.Device("llvmpipe")` or `Mantle.Device("radeon")`.

## Million token context cost

For full-attention layers, the FP16 KV storage is:

`tokens * full_attention_layers * 2 * kv_heads * head_dimension * 2 bytes`

The Q8 estimate below adopts BonsaiRunner's layout: Int8 K and V, with one Float32 scale per head per token for each. It is a storage estimate, not proof that this quantization preserves Qwen's accuracy over a million positions. [inspect_models.jl](../../../tools/small_llm/inspect_models.jl) downloads pinned configurations and reproduces these calculations; [model_inventory.json](../../../tools/small_llm/model_inventory.json) contains the results.

| Model | FP16 KV at 1M tokens | Q8 KV at 1M tokens | Q8 cache scan alone at assumed 220 GB/s |
| --- | ---: | ---: | ---: |
| Qwen3.5-2B | 11.44 GiB | 5.81 GiB | 28.4 ms/token, at most 35.3 tokens/s |
| Qwen3.5-4B | 30.52 GiB | 15.50 GiB | 75.6 ms/token, at most 13.2 tokens/s |
| Qwen3.6-35B-A3B | 19.07 GiB | 9.69 GiB | 47.3 ms/token, at most 21.2 tokens/s |

These illustrative bounds assume one full read of the stored cache per generated token at the stated effective bandwidth. They omit weights, recurrent state, dequantization and compute; they are not hardware-wide impossibility proofs. Full-attention prefill also has quadratic attention arithmetic, even though FlashAttention avoids materializing the quadratic score matrix. Hybrid recurrent layers reduce the number of full-attention layers without eliminating their cost.

The 4B recurrent matrices contribute about 48 MiB, plus convolution history; attention cache dominates at long context. At 32K tokens, its estimated Q8 KV cache is only about 0.51 GiB. This motivates measuring the short working context and million-token setting separately.

Implement the documented YaRN scaling rather than merely increasing an allocation limit. Evaluate both 1,000,000 total tokens and sufficient output headroom under the documented 1,010,000 limit. Test multi-fact recall, binding, negation, conflicting updates and long chains at multiple positions. A single needle lookup is insufficient. [Lost in the Middle](https://arxiv.org/abs/2307.03172) motivates position-sensitive tests; [LongMemEval](https://arxiv.org/abs/2410.10813) motivates temporal updates and abstention.

Retain three distinct comparison conditions: full million-token attention, a million-token external store with retrieved text, and a million-token store accessed through latent memory. They answer different questions; an external store does not satisfy the full active-context requirement by definition.

## Experiments with the best chance of teaching us something

The individual components have substantial prior art. The following combinations are useful hypotheses for this engine; their novelty has not been established by this survey.

1. **Automatic evidence during decoding.** Trigger a language-server or solver monitor from parser state and unresolved references, and compare it with explicit model-authored tool queries. This directly tests whether the weak model's query formulation is the bottleneck. Use an oracle-retrieval condition to distinguish retrieval failures from failures to use evidence.
2. **The same evidence through three interfaces.** Supply an identical API signature, graph neighborhood or proof result as text, as a probability adjustment, or as trained latent memory. Hold evidence content, search budget and wall time constant. This is the cleanest test of the central hypothesis about influencing thinking.
3. **A learned bridge from exact solver state.** Train a small encoder and cross-attention adapter on generated Datalog/Prolog proofs and finite Bayesian networks. Evaluate new entity names, rule combinations, proof depths and probability values. Measure exact answers, calibration and counterfactual updates rather than fluent explanations.
4. **Procedural add-ons with replaceable facts.** Give each language module parser/type/LSP state and optionally a trained adapter; share the reasoning backbone. Swap API versions and factual stores during evaluation. Test mixed-language tasks to see whether modules compose.
5. **Adaptive intervention under a latency budget.** Query expensive systems at identifier boundaries, proof steps or uncertain decisions. Cache results by semantic state and source version. Compare a learned scheduler with fixed schedules and oracle intervention. Entropy alone does not reliably identify an error: a wrong prediction can be confident.

A minimal probability interface can mix a neural token distribution with a retrieved token distribution, then restrict verified invalid continuations. A product-of-experts alternative adds calibrated evidence scores to logits. The latter is not automatically Bayesian: sources may be correlated, and a posterior already contains a prior. Avoid multiplying the same prior or repeated evidence twice. Multi-token facts and constraints require token-aware tries, incremental parsers, or branching over complete candidate spans.

For exact logic, represent goals, bindings and finite candidate actions explicitly. A missing fact is not necessarily false, and an incomplete LSP response must not become a hard exclusion mask. Distinguish unknown, false, solver timeout and unsupported syntax. Use hard masks only when the monitor can establish impossibility under its stated semantics. Otherwise apply a soft score or allow the model to seek more evidence.

Measure task success per wall-clock second, prompt ingestion, time to first token, decode throughput at several cache lengths, p50/p95 latency, memory, solver cost, retrieval quality, compilation rate, numerical calibration, and factual-update consistency. Compare against the same small model without augmentation, ordinary tool/RAG loops, long-context prompting, and a stronger local baseline. A faster token generator that needs many failed attempts can still be the slower problem solver.

## Reproducible local checks

The candidate inventory is generated from revision-pinned upstream configurations. It validates the layer count and calculates KV and recurrent-matrix storage. It downloads no model weights and executes no remote model code.

The fresh baseline harness is [bench_bonsai.jl](../../../tools/small_llm/bench_bonsai.jl). It uses the real existing Mantle/DNNKernels runner, records repeated fixed-length decode samples and GPU clock readings, asserts finite logits and deterministic token sequences, and closes sessions explicitly. Its result file, when the run succeeds, is `tools/small_llm/bonsai_baseline.json`. It measures short-context throughput; retaining EOS tokens makes this a fixed-work runtime measurement rather than an answer-quality benchmark.

Use the workspace root environment, include the inspection script and call `inspect_models()`. For the compute-only GPU baseline, set `ENV["JULIA_GLFW_PLATFORM"] = "null"` before importing Mantle, include the benchmark script, and call `bench_bonsai("gen/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf"; output="baseline.json")`. Mantle imports GLFW even for this compute workload; GLFW's supported null platform avoids requiring an authenticated X11 display. The initial baseline reached model loading and prefill but failed on an ambiguous unqualified `step!` exported by both BonsaiRunner and DNNKernels. The harness now explicitly calls `BonsaiRunner.step!`.

Qwen throughput through occupied 4K context and initial reference/feedback parity are now measured. Occupied 32K context, full million-token ingestion and recall, a trained latent interface, and any general reasoning improvement need separate evidence. The next decision gate is efficient prefill and longer-context throughput followed by controlled evidence-interface comparisons.
