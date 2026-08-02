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

[`assetpath`](@ref) below is what remains of that, and it has exactly one class
of caller left — the runner packages for models that are **not ported yet**, which
have no artifact to bind because they have no export to bind. Each one stops
using it at the moment it is ported.

Walking up rather than a fixed `../../../gen` because the depth of a generated
tree above a package is a property of the checkout, not of the package: the
fixed form silently broke every runner the day they moved into a monorepo, and
it is wrong on its face now that the monorepo is also a standalone repository
where no such tree exists.

Nothing here throws when an asset is missing. Every workload guards on
`isdir`/`isfile` and precompiles nothing rather than failing, so a fresh clone
with no weights installs and loads — it just has no model until they are.
"""

using Artifacts
using LazyArtifacts

"""
    findasset(relative; env = nothing, from = @__DIR__) -> String

The first existing `<ancestor>/<relative>` walking up from `from`, or — if none
exists — the path relative to the outermost ancestor tried, so a caller has
something concrete to name in an error.

`env` names an environment variable that wins outright when set.
"""
function findasset(relative::AbstractString; env::Union{Nothing,AbstractString} = nothing,
                   from::AbstractString = @__DIR__)
    if env !== nothing
        p = get(ENV, env, "")
        isempty(p) || return p
    end
    dir = normpath(from)
    last = dir
    while true
        candidate = joinpath(dir, relative)
        ispath(candidate) && return candidate
        parent = dirname(dir)
        parent == dir && break          # reached the filesystem root
        last = dir = parent
    end
    return joinpath(last, relative)
end

"""
    assetpath(; generated, env, from) -> String

A model's *locally provided* asset directory: the environment override if it is
set, else the generated tree this checkout would have.

  * `generated` — the path, relative to some ancestor, that a machine which
    produces these files would have (e.g. `gen/graphs/sam2-large`).
  * `env` — an environment variable that overrides everything.
  * `from` — where to start walking up; pass `@__DIR__` from the caller, since
    `@__DIR__` here would be this package rather than the one asking.

The returned path may not exist, and that is deliberate: the caller names it in
the error so the message says where it looked.

**Only for models that are not ported yet.** A ported one writes
`assetdir() = @artifact_str("name")` and does not come here at all — see the
module docstring.
"""
function assetpath(; generated::AbstractString,
                   env::Union{Nothing,AbstractString} = nothing,
                   from::AbstractString = @__DIR__)
    if env !== nothing
        p = get(ENV, env, "")
        isempty(p) || return p
    end
    return findasset(generated; from)
end
