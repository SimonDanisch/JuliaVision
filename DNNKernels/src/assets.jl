"""
Finding a model's graphs and weights.

They are not source, and they are far too large to be: SAM 2.1's weights alone
are 942 MB. So they are Julia *artifacts* — content-addressed, downloaded on
first use, shared between every environment on the machine, and never in git.

**A ported model reads its assets from its artifact and from nowhere else:**

    assetdir() = @artifact_str("sam2-large")

That is the whole mechanism. `artifact"..."` downloads on first use, verifies the
tree hash, caches across every environment on the machine and hands back a path.
Each package writes the literal itself, because `@artifact_str` resolves the
`Artifacts.toml` next to the module that expands it, which is the one that binds
it.

**Changing a model's assets means re-binding its artifact.** Re-export, then
`tools/make_artifacts.jl <name>`: that hashes the new tree and rewrites the
`Artifacts.toml`, so `assetdir()` resolves to the new content immediately, and
uploading is only how it reaches anyone else. There is deliberately no "use the
working copy instead" path — one source of truth, and the way to change it is to
change it.

Two earlier designs are worth not repeating. The first wrapped the download in an
`artifactpath(name, toml)` calling `Artifacts.ensure_artifact_installed`, which
does not exist — downloading needs `LazyArtifacts` or `Pkg`. Every lazy artifact
raised `UndefVarError` into a `catch` that logged a warning and returned
`nothing`. The second, fixing that, still consulted an environment variable and a
locally generated tree first. Nothing ever set the environment variables, and the
generated-tree branch is what made the broken download invisible for as long as
it was: on a machine that has a `gen/` tree, the fallback always answered.

There is no second path left. A package for a model that is not ported yet does
not fall back to a directory either — its `assetdir()` says so and throws, so the
message a caller gets names the port, not a path that will never exist.

A fresh clone still installs and loads with no assets anywhere: every workload
guards on its package's `ready()` and precompiles nothing rather than failing.
What changed is that a *not ported* package now says so through `assetdir()`
instead of handing back a path nobody will ever have.

**This applies to tests too, which is where it had survived.** "There is no
second path left" was written about the runner packages and was not true of the
test suite: `findasset` walked up the filesystem looking for a `gen/` tree, with
an environment variable that won outright — the same two mechanisms this
docstring says were removed, one directory over. It has been deleted.

The consequence is worth stating because it is the whole point. A test whose
assets come from a local `gen/` tree passes for whoever ran the exporter and is
unreachable for everyone else, so the layer-by-layer PyTorch parity gate — the
one check that catches a kernel which is fast and subtly wrong — ran only on the
machine that generated it. Assets that only tests read get their own lazy
artifact, the way `sam2-large-refs` sits beside `sam2-large`, so the gate travels
with the repository instead of with one checkout.
"""

using Artifacts
using LazyArtifacts


