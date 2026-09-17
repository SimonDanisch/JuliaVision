# `aten::lstm.input`, declared, against torch's definition written out by hand.
#
# The kernel is shared with the interpreted path, so comparing the two routes
# cannot check the RECURRENCE — it checks the plumbing around it and both arms
# would be wrong together. What is checked here is the arithmetic itself, from
# the definition in torch's docs: gate order `i, f, g, o`, the reverse direction
# reading the sequence backwards but writing each result at its own timestep.
#
# In the UNSATURATED regime, deliberately. With weights large enough to saturate
# the gates every output is `±1` and the test passes no matter which row a
# direction wrote, which timestep it read, or whether `h` carried at all. The
# weights below are scaled so the outputs land in `±0.4`; the assertion at the
# end pins that, so a future edit cannot quietly make the test vacuous again.
#
# Declared, this op is four passes per direction — `w_ih` transposed, the input
# projection as one GEMM over the whole sequence, `w_hh` transposed, then the
# recurrence — and any of them landing in the wrong buffer shows up here.

using Test, Random
import DNNKernels
import Mantle

const DK = DNNKernels
const MM = Mantle

"""
One direction of a single-layer LSTM, on the host, from the definition.

`x` is `(D, T)` and the result `(H, T)`. `w_ih` is `(D, 4H)` and `w_hh` is
`(H, 4H)`, which is torch's `(4H, D)` / `(4H, H)` in the reversed layout, so the
products are transposed here rather than in the caller.
"""
function hostlstm(x, w_ih, w_hh, b_ih, b_hh, h0, c0; rev::Bool = false)
    D, T = size(x)
    H = size(w_hh, 1)
    out = zeros(Float32, H, T)
    h, c = copy(h0), copy(c0)
    for step in 1:T
        t = rev ? T - step + 1 : step
        gates = transpose(w_ih) * x[:, t] .+ b_ih .+ transpose(w_hh) * h .+ b_hh
        i = @. 1 / (1 + exp(-gates[1:H]))
        f = @. 1 / (1 + exp(-gates[(H + 1):(2H)]))
        z = tanh.(gates[(2H + 1):(3H)])
        o = @. 1 / (1 + exp(-gates[(3H + 1):(4H)]))
        c = @. f * c + i * z
        h = @. o * tanh(c)
        out[:, t] = h
    end
    return out
end

"""
A one-op ATen graph holding a bidirectional `lstm.input`.

The state and the eight parameters are `:weight` buffers so the test supplies
them by value; the op names them through `attrs["arg1"]`/`["arg2"]` exactly as
the export does, which is also the ordering contract the emit reads. The result
is a THREE-element tuple with a `getitem` view on element 0, because that is the
shape a composite op's result has and the shape `resultshapes` reasons about.
"""
function lstmgraph(D, H, T)
    buf(id, kind, shape, key = "", of = "", vop = "",
        attrs = Dict{String,Any}()) =
        DK.Buffer(id, kind, shape, Float32, key, (0, 2), of, vop, attrs)
    names = ["wih0", "whh0", "bih0", "bhh0", "wih1", "whh1", "bih1", "bhh1"]
    bufs = Dict{String,DK.Buffer}()
    bufs["x"] = buf("x", :external, Any[1, T, D])
    bufs["h0"] = buf("h0", :weight, Any[2, 1, H], "h0")
    bufs["c0"] = buf("c0", :weight, Any[2, 1, H], "c0")
    for p in names
        shape = startswith(p, "b") ? Any[4H] :
                endswith(p[1:3], "ih") ? Any[4H, D] : Any[4H, H]
        bufs[p] = buf(p, :weight, shape, p)
    end
    bufs["ls"] = buf("ls", :transient, Any[], "", "", "",
                     Dict{String,Any}(
                         "shapes" => Any[Any[1, T, 2H], Any[2, 1, H], Any[2, 1, H]],
                         "dtypes" => Any[Float32, Float32, Float32]))
    bufs["y"] = buf("y", :view, Any[1, T, 2H], "", "ls",
                    "<built-in function getitem>",
                    Dict{String,Any}("arg1" => 0))
    attrs = Dict{String,Any}("arg1" => ["\$h0", "\$c0"],
                             "arg2" => ["\$" * p for p in names],
                             "arg3" => true, "arg4" => 1, "arg5" => 0,
                             "arg6" => false, "arg7" => true, "arg8" => true)
    op = DK.Op("ls", "lstm.input", vcat(["x", "h0", "c0"], names), "ls", attrs)
    order = vcat(["x", "h0", "c0"], names, ["ls", "y"])
    return DK.Graph("lstm", String[], ["x"], ["y"], bufs, order, [op])
end

"""Small weights, so the gates stay off their rails."""
function lstmweights(D, H)
    Random.seed!(7)
    w = Dict{String,Any}()
    for p in ("wih0", "wih1"); w[p] = 0.3f0 .* randn(Float32, D, 4H); end
    for p in ("whh0", "whh1"); w[p] = 0.3f0 .* randn(Float32, H, 4H); end
    for p in ("bih0", "bih1", "bhh0", "bhh1"); w[p] = 0.2f0 .* randn(Float32, 4H); end
    w["h0"] = 0.1f0 .* randn(Float32, H, 1, 2)
    w["c0"] = 0.1f0 .* randn(Float32, H, 1, 2)
    return w
end

function declaredlstm(be)
@testset "a declared bidirectional LSTM matches the definition" begin
    D, H, T = 6, 4, 5
    w = lstmweights(D, H)
    Random.seed!(11)
    xh = 0.5f0 .* randn(Float32, D, T, 1)
    g = lstmgraph(D, H, T)
    decl = DK.declaredvalues(g, Dict{String,Any}("x" => xh), w;
                             dims = (;), backend = be)
    got = reshape(Array(DK.tohost(decl["y"])), 2H, T)
    fwd = hostlstm(xh[:, :, 1], w["wih0"], w["whh0"], w["bih0"], w["bhh0"],
                   w["h0"][:, 1, 1], w["c0"][:, 1, 1]; rev = false)
    bwd = hostlstm(xh[:, :, 1], w["wih1"], w["whh1"], w["bih1"], w["bhh1"],
                   w["h0"][:, 1, 2], w["c0"][:, 1, 2]; rev = true)
    want = vcat(fwd, bwd)
    @test size(got) == size(want)
    @test maximum(abs.(Float64.(got) .- Float64.(want))) <=
          1e-5 * maximum(abs, want)
    # The reverse half is not the forward half: if `rowoff` or `REV` were wrong
    # the two blocks would agree, and so would the test.
    @test maximum(abs.(Float64.(fwd) .- Float64.(bwd))) > 1e-2
    # And nothing saturated, so the comparison above has something to see.
    @test maximum(abs, want) < 0.9
    @test maximum(abs, want) > 0.05
end
end

# A graph that asks for the final `h` or `c` is REFUSED rather than handed
# memory nothing wrote: `lstm_kernel!` keeps both in shared memory for the whole
# loop and writes out only the per-step output.
function lstmstate(be)
@testset "the LSTM's final state is refused, not fabricated" begin
    D, H, T = 6, 4, 5
    g = lstmgraph(D, H, T)
    # Element 1 is the final `h`; give it a reader.
    bufs = Dict{String,DK.Buffer}(g.buffers)
    bufs["hn"] = DK.Buffer("hn", :view, Any[2, 1, H], Float32, "", (0, 2), "ls",
                           "<built-in function getitem>",
                           Dict{String,Any}("arg1" => 1))
    g2 = DK.Graph(g.name, g.symbols, g.inputs, ["y", "hn"], bufs,
                  vcat(g.order, ["hn"]), g.ops)
    err = try
        DK.declaredvalues(g2, Dict{String,Any}("x" => zeros(Float32, D, T, 1)),
                          lstmweights(D, H); dims = (;),
                          backend = be)
        nothing
    catch e
        e
    end
    @test err !== nothing
    @test occursin("final h", sprint(showerror, err))
end
end

let bes = MM.eachbackend()
    isempty(bes) && @info "no usable backend; skipping the declared LSTM"
    for be in bes
        declaredlstm(be)
        lstmstate(be)
    end
end
