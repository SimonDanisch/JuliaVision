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
 3. **The artifact**, downloaded if `Artifacts.toml` binds one.

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
    artifactpath(name, toml) -> String or nothing

Where artifact `name` is installed, downloading it if `Artifacts.toml` binds it
lazily; `nothing` when it is not bound at all.

Returns `nothing` rather than throwing because an unbound artifact is the normal
state of a working checkout — the assets are generated there, not fetched — and
[`assetpath`](@ref) has already tried the generated tree by the time it asks.
"""
function artifactpath(name::AbstractString, toml::AbstractString)
    isfile(toml) || return nothing
    meta = Artifacts.artifact_meta(name, toml)
    meta === nothing && return nothing
    hash = Base.SHA1(meta["git-tree-sha1"])
    try
        Artifacts.artifact_exists(hash) ||
            Artifacts.ensure_artifact_installed(name, meta, toml)
        return Artifacts.artifact_path(hash)
    catch err
        # A download that fails must not take the whole session with it: the
        # caller can still be pointed at a local copy, and every workload
        # tolerates the asset being absent.
        @warn "could not install artifact $name" exception = err
        return nothing
    end
end

"""
    assetpath(; artifact, toml, generated, env, from) -> String

A model's asset directory, by the three-step rule this file documents.

  * `artifact` — the artifact name to fall back on, or `nothing` for none.
  * `toml` — the `Artifacts.toml` that binds it, normally the package's own.
  * `generated` — the path, relative to some ancestor, that a machine which
    produces these files would have (e.g. `gen/graphs/sam2-large`).
  * `env` — an environment variable that overrides everything.
  * `from` — where to start walking up; pass `@__DIR__` from the caller, since
    `@__DIR__` here would be this package rather than the one asking.

The returned path may not exist. That is deliberate — see the module docstring.
"""
function assetpath(; artifact::Union{Nothing,AbstractString} = nothing,
                   toml::AbstractString = "",
                   generated::AbstractString,
                   env::Union{Nothing,AbstractString} = nothing,
                   from::AbstractString = @__DIR__)
    if env !== nothing
        p = get(ENV, env, "")
        isempty(p) || return p
    end
    local_ = findasset(generated; from)
    ispath(local_) && return local_
    if artifact !== nothing
        p = artifactpath(artifact, toml)
        p === nothing || return p
    end
    return local_        # name the place we looked, so the error can say it
end
