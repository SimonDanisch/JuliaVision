"""
Fold a functional KV-cache update into an in-place write.

An exported decode step cannot say "write into the cache I was given". It says

    select_i   = select(k_cache, 0, i)            # a view, per layer
    updated_i  = index_put(select_i, position, k) # a COPY of the whole layer
    k_cache'   = cat(updated_0 .. updated_63)     # a COPY of the whole cache

and both copies are real. On K2 Horizon 32B at a 1024-slot cache that is 128
`index_put`s of 2 MiB and two 128 MiB stacks — about 1 GiB of pure copying per
token, 11.3 ms of a 190 ms step, and it grows with the cache. At 4096 slots it
was 67 ms.

Nothing about it is necessary. Each layer writes a DIFFERENT slice, so the
writes cannot alias each other; the only reader of `select_i` is its own
`index_put`; and the `cat`'s only consumer is a graph output that the caller
hands straight back on the next step. So the write can go into `k_cache` itself
and the `cat` is the identity.

What the pass checks before rewriting, because each of these is what makes it
safe rather than merely plausible:

  * the `cat` concatenates exactly `size(X, 1)` values along dim 0, in order,
  * every one of them is an `index_put` whose target is `select(X, 0, i)`,
  * `X` is a graph INPUT, so the caller owns it and expects it written,
  * nothing else in the graph reads `X` or any of its views, so no op can
    observe the pre-write contents,
  * the `cat`'s output leaves the graph rather than being read again.

A graph that fails any of them keeps its copies.
"""

"""
    foldcacheupdate(graphs) -> (graphs, n)

Rewrite `cat(index_put(select(X, 0, i), …) for i)` into in-place writes on `X`.
`n` counts the caches folded.
"""
function foldcacheupdate(graphs::AbstractDict)
    out = Dict{String,Any}()
    total = 0
    for (k, g) in graphs
        if g isa Graph
            g2, n = foldcacheupdate(g)
            out[k] = g2; total += n
        else
            out[k] = g
        end
    end
    (out, total)
end

function foldcacheupdate(g::Graph)
    producer = Dict{String,Op}()
    for op in g.ops
        isempty(string(op.out)) || (producer[op.out] = op)
    end
    # Which buffers are views of which root, and who reads what.
    readers = Dict{String,Int}()
    for op in g.ops, id in op.ins
        readers[viewroot(g, id)] = get(readers, viewroot(g, id), 0) + 1
    end

    ops = copy(g.ops)
    buffers = Dict{String,Buffer}(g.buffers)
    folded = 0
    for (i, c) in enumerate(ops)
        c.aten == "cat.default" || continue
        # cat along dim 0 (torch), which is the default when no axis is given.
        Int(get(c.attrs, "arg1", 0)) == 0 || continue
        n = length(c.ins)
        n > 1 || continue

        # Every input an `index_put` on a `select(X, 0, j)`, in order, same X.
        X = ""
        ok = true
        puts = Op[]
        for (j, id) in enumerate(c.ins)
            p = get(producer, id, nothing)
            if p === nothing || p.aten != "index_put.default" || isempty(p.ins)
                ok = false; break
            end
            b = get(g.buffers, p.ins[1], nothing)
            if b === nothing || b.kind !== :view || b.viewop != "select.int" ||
               Int(get(b.attrs, "arg1", -1)) != 0 || Int(get(b.attrs, "arg2", -1)) != j - 1
                ok = false; break
            end
            j == 1 ? (X = String(b.of)) : (String(b.of) == X || (ok = false))
            ok || break
            push!(puts, p)
        end
        ok && !isempty(X) || continue

        xb = get(buffers, X, nothing)
        xb === nothing && continue
        # A graph INPUT: the caller owns the storage and gets it back written.
        (X in g.inputs || xb.kind === :external) || continue
        # Exactly one slice per leading index, and no other reader of X.
        length(c.ins) == Int(evalconst(xb.shape[1])) || continue
        get(readers, X, 0) == n || continue
        # The result must leave rather than be read again.
        escapes(g, c.out) || continue

        for p in puts
            ops[findfirst(==(p), ops)] = Op(p.id, "index_put.default", p.ins, p.out,
                                            merge(p.attrs, Dict{String,Any}("inplace" => true)))
        end
        # The stack is now the identity on X.
        ops[i] = Op(c.id, "alias.default", [X], c.out, Dict{String,Any}())
        folded += 1
    end
    folded == 0 && return (g, 0)
    (Graph(g.name, g.symbols, g.inputs, g.outputs, buffers, g.order, ops, g.fusion), folded)
end

"""Whether `id` (or a view of it) is a declared output of `g`."""
function escapes(g::Graph, id::AbstractString)
    id in g.outputs && return true
    for o in g.outputs
        b = get(g.buffers, o, nothing)
        b === nothing && continue
        viewroot(g, o) == id && return true
    end
    false
end

"""A shape entry that must be an integer by now."""
evalconst(s) = s isa Integer ? Int(s) : error("foldcacheupdate: symbolic extent $s")
