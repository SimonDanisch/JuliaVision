"""
Flow-matching Euler sampling: the schedule, the step and classifier-free guidance,
shared by every diffusion model on this runtime.

All of them are diffusers' `FlowMatchEulerDiscreteScheduler` with different
parameters, so this is that scheduler and each runner states only its own
parameters. Three models used to carry their own copies, and two of the copies
had drifted from the reference: Qwen-Image's schedule differed in the last bit
for a third of image sizes, and Hunyuan3D's guidance rounded its scale to fp16
before multiplying. `test/test_flowmatch.jl` pins all of it against checksums
taken from diffusers itself.

**Rounding order is the contract here, not the formula.** Each function below
reproduces where PyTorch and numpy round, and on a GPU that depends on how the
code is split into broadcasts: Lava keeps a fused fp16 expression in fp32 and
rounds once at the store, while PyTorch rounds after every op. Where the
reference rounds an intermediate, the code materialises it.
"""

"""
    linspace(start, stop, n) -> Vector{Float64}

`numpy.linspace`, rounding the way numpy does: `k * step + start`, with the last
element pinned to `stop`. Not Julia's `range`, which rounds each element
correctly and so lands one Float64 ULP away often enough to move a Float32
schedule built from it.
"""
function linspace(start::Float64, stop::Float64, n::Integer)
    n == 1 && return [start]
    step = (stop - start) / (n - 1)
    return [k == n - 1 ? stop : k * step + start for k in 0:(n - 1)]
end

"""
How the scheduler warps its raw sigmas. `NoShift`, `StaticShift(shift)` (the
config's `shift`), or `ExponentialShift(mu)` (`use_dynamic_shifting` with
`time_shift_type = "exponential"`, `mu` from [`calculate_shift`](@ref)).

The parameters are Float64 because they are Python floats upstream; numpy rounds
them to float32 where they meet the float32 sigmas, and [`shiftsigmas`](@ref)
does the same.
"""
abstract type FlowShift end
struct NoShift <: FlowShift end
struct StaticShift <: FlowShift
    shift::Float64
end
struct ExponentialShift <: FlowShift
    mu::Float64
end

shiftsigmas(::NoShift, s::Vector{Float32}) = s

function shiftsigmas(x::StaticShift, s::Vector{Float32})
    k, km1 = Float32(x.shift), Float32(x.shift - 1)
    return @. k * s / (1 + km1 * s)
end

function shiftsigmas(x::ExponentialShift, s::Vector{Float32})
    e = Float32(exp(x.mu))
    return @. e / (e + (1 / s - 1))
end

"""
    calculate_shift(seqlen; base_seq_len = 256, max_seq_len = 4096,
                    base_shift = 0.5, max_shift = 1.15) -> Float64

The `mu` of an [`ExponentialShift`](@ref): linear in the image's token count,
from `base_shift` at `base_seq_len` to `max_shift` at `max_seq_len`. Defaults
are diffusers' (FLUX's); a model passes its scheduler config's values.
"""
function calculate_shift(seqlen::Integer; base_seq_len::Integer = 256,
                         max_seq_len::Integer = 4096, base_shift::Float64 = 0.5,
                         max_shift::Float64 = 1.15)
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return seqlen * m + b
end

"""
    flowschedule(sigmas, shift; terminal = 0f0, shiftterminal = nothing,
                 trainsteps = 1000) -> (; sigmas, timesteps)

`FlowMatchEulerDiscreteScheduler.set_timesteps(sigmas = ...)`. `sigmas` are the
raw levels a pipeline passes (Float64, as numpy builds them, usually with
[`linspace`](@ref)); they are rounded to Float32, warped by `shift`, stretched so
the last one lands on `shiftterminal` if that is given, and then `terminal` is
appended. The result has one more sigma than timesteps: step `k` goes from
`sigmas[k]` to `sigmas[k + 1]` and conditions the model on `timesteps[k]`, which
is the sigma on the `trainsteps` scale.

`terminal` is 0 for a schedule that runs from noise to data. Hunyuan3D's runs the
other way and appends 1.
"""
function flowschedule(sigmas::AbstractVector{Float64}, shift::FlowShift;
                      terminal::Float32 = 0f0,
                      shiftterminal::Union{Nothing,Float64} = nothing,
                      trainsteps::Integer = 1000)
    s = shiftsigmas(shift, Float32.(sigmas))
    if shiftterminal !== nothing
        omz = 1 .- s
        scale = omz[end] / Float32(1 - shiftterminal)
        s = @. 1 - omz / scale
    end
    return (; sigmas = [s; terminal], timesteps = s .* Float32(trainsteps))
end

"""
    eulerstep(x, v, sigma, sigma_next) -> x′
    eulerstep!(x, v, sigma, sigma_next) -> x

One Euler step of the flow ODE, as `scheduler.step` computes it:

    prev_sample = sample.to(float32) + (sigma_next - sigma) * model_output
    prev_sample = prev_sample.to(model_output.dtype)

**The step size is narrowed to the prediction's dtype before the multiply.**
`sigma_next - sigma` is a 0-dimensional float32 tensor against a dimensioned fp16
one, and PyTorch's promotion lets the dimensioned operand decide, so the product
is fp16 with the scalar rounded to fp16 first. At 50 steps that is
`Float16(1/49) = 0.0204` against 0.020408163, taken every step. Only the sum with
the sample is float32. The product is its own broadcast so a GPU rounds it where
PyTorch does.

`x` and `v` share an element type, as they do in every pipeline here; either may
be a `Mantle.Buffer`.
"""
eulerstep(x, v, sigma::Float32, sigma_next::Float32) =
    eulerstep(M.storage(x), M.storage(v), sigma, sigma_next)

function eulerstep(x::AbstractArray{T}, v::AbstractArray{T}, sigma::Float32,
                   sigma_next::Float32) where {T}
    step = T(sigma_next - sigma) .* v
    return T.(Float32.(x) .+ Float32.(step))
end

eulerstep!(x, v, sigma::Float32, sigma_next::Float32) =
    (eulerstep!(M.storage(x), M.storage(v), sigma, sigma_next); x)

function eulerstep!(x::AbstractArray{T}, v::AbstractArray{T}, sigma::Float32,
                    sigma_next::Float32) where {T}
    axes(x) == axes(v) || throw(DimensionMismatch(
        "sample $(size(x)) and prediction $(size(v)) differ in shape"))
    step = T(sigma_next - sigma) .* v
    x .= T.(Float32.(x) .+ Float32.(step))
    return x
end

# Arrays of two element types. Without these the unwrapping methods above would
# take them, unwrap nothing and call themselves.
eulerstep(x::AbstractArray, v::AbstractArray, ::Float32, ::Float32) = throw(mixeddtypes(x, v))
eulerstep!(x::AbstractArray, v::AbstractArray, ::Float32, ::Float32) = throw(mixeddtypes(x, v))
mixeddtypes(x, v) = ArgumentError("sample is $(eltype(x)) and prediction $(eltype(v)); " *
                                  "the step rounds to one dtype, so a pipeline keeps both in it")

"""
    cfg(cond, uncond, scale) -> v

Classifier-free guidance, `uncond + scale * (cond - uncond)`, rounded as PyTorch
rounds it: the difference and the scaled difference are each rounded to the
predictions' dtype, and the scale itself is not. A Python-float scale stays
float32 inside the multiply ("opmath"), so rounding it to fp16 first is a
different number whenever the scale is not an fp16 value: at 3.3 that moved 5 of
16 outputs. Three broadcasts, so a GPU rounds between them as PyTorch does.
"""
function cfg(cond::AbstractArray{T}, uncond::AbstractArray{T}, scale::Real) where {T}
    d = cond .- uncond
    d = T.(Float32(scale) .* Float32.(d))
    return uncond .+ d
end
