"""
Finding a model's graphs and weights.

They are not source, and they are far too large to be: SAM 2.1's weights alone
are 942 MB. So they are Julia *artifacts* — content-addressed, downloaded on
first use, shared between every environment on the machine, and never in git.

Three places are consulted, in this order, and the order is the design:

 1. **An environment variable**, when the caller has them somewhere specific.
    Precompilation has no running program to ask, so this is how a workload is
    pointed at a checkout that is not the default one.
 2. **A locally generated tree**, found by walking up from the package. On a
    machine that *produces* these files — anything with `tools/export_sam2.py`
    and a PyTorch checkout — the generated ones are the interesting ones, and
    downloading a published copy over the top of work in progress would be
    exactly wrong.
 3. **The artifact** — and that step is not implemented here. `artifact"name"`
    already downloads on first use, verifies the tree hash, caches across every
    environment on the machine and hands back a path. Each package writes that
    literal itself, because `@artifact_str` resolves the `Artifacts.toml` next to
    the module that expands it, which is the one that binds the artifact.

This file used to wrap step 3 in an `artifactpath(name, toml)` that called
`Artifacts.ensure_artifact_installed` — which does not exist; downloading needs
`LazyArtifacts` or `Pkg`. Every lazy artifact therefore raised `UndefVarError`
into a `catch` that logged "could not install artifact" and returned `nothing`,
so every caller silently fell through to the generated tree. On a developer
machine, which always has one, that is invisible. The lesson is the reason the
wrapper is gone rather than fixed: `@artifact_str` is the supported path and it
does not need helping.

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

The returned path may not exist, and that is deliberate: a package with no
artifact names it in the error so the message says where it looked.

The artifact is deliberately not a case here. A package that binds one writes

    assetdir() = (p = assetpath(...); ispath(p) ? p : artifact"name")

and `artifact"..."` does the downloading, hashing and caching — see the module
docstring for why this file no longer tries to.
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
