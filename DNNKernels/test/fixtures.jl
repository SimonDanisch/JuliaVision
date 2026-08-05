"""
Graphs for the pass tests, from this package's own artifact bindings.

`foldoutcasts`, `constfold`, `plan` and `diagnostics` test **DNNKernels**, and a
test belongs with its subject — but each needs a real exported graph, because a
synthetic one does not exercise a fold or a plan the way a 400-op encoder does.
Five test files were importing `SAM2Runner`/`MatAnyoneRunner` to get one, which
made the kernel library's suite unrunnable without a model package installed.

`DNNKernels/Artifacts.toml` binds the same two artifacts the runners bind. They
are content-addressed, so identical `git-tree-sha1`s resolve to one directory in
`~/.julia/artifacts` — a binding, not a download.

Graphs only. Weights, references and the layer-by-layer parity test stay with
their models; this is the minimum a pass test needs to run against something real.
"""
module Fixtures

using Artifacts, LazyArtifacts
using DNNKernels: loadgraph

"The MatAnyone graph named `n` (`transform_key`, `encode_image`, …)."
matanyone(n::AbstractString) = loadgraph(joinpath(artifact"matanyone", "graphs", "$n.json"))

"SAM 2's `sam2_encoder` or `sam2_decoder`."
sam2(n::AbstractString) = loadgraph(joinpath(artifact"sam2-large", "$n.json"))

"Names present in the artifact, so a test can skip precisely rather than throw."
matanyonenames() = [splitext(f)[1] for f in readdir(joinpath(artifact"matanyone", "graphs"))
                    if endswith(f, ".json")]
have(n::AbstractString) = n in matanyonenames()

end
