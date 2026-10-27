# Never evaluate generated code here. No benchmark tests/expected outputs enter
# this analyzer: syntax + the already-published dependency contract only.
module PublicProgramDiagnostics
const ALLOWED=Set([:Base,:Core,:Main,:JSON3,:CairoMakie,:Makie,:Base64,:Statistics,
    :LinearAlgebra,:Random,:Dates,:Printf,:DelimitedFiles,:SparseArrays,:Serialization,
    :SHA,:Sockets,:UUIDs,:Unicode,:Test,:TOML,:Logging,:Markdown,:InteractiveUtils,
    :Libdl,:Mmap,:Distributed,:SharedArrays,:REPL,:Pkg,:Downloads,:FileWatching,
    :Future,:Artifacts,:LazyArtifacts,:LibGit2,:NetworkOptions,:SuiteSparse,:Tar])

function import_root(expression)
    expression isa Symbol && return expression===Symbol(".") ? nothing : expression
    expression isa Expr || return nothing
    expression.head in (:as,Symbol(":")) && return import_root(first(expression.args))
    if expression.head===Symbol(".")
        isempty(expression.args)&&return nothing
        first(expression.args)===Symbol(".")&&return nothing # relative module
        return import_root(first(expression.args))
    end
    nothing
end

function analyze(code::AbstractString)
    ncodeunits(code)<=262144 || throw(ArgumentError("bounded source required"))
    tree=Meta.parseall(code);stack=Any[tree];parse_errors=String[];imports=Symbol[]
    visited=0
    while !isempty(stack)
        node=pop!(stack);node isa Expr || continue
        visited+=1;visited<=100000 || throw(ArgumentError("bounded syntax tree required"))
        if node.head in (:error,:incomplete)
            detail=isempty(node.args) ? string(node.head) : sprint(showerror,first(node.args))
            push!(parse_errors,first(detail,min(length(detail),4096)))
        elseif node.head in (:using,:import)
            for expression in node.args
                root=import_root(expression);root===nothing||push!(imports,root)
            end
        elseif node.head in (:toplevel,:block,:if,:elseif,:try)
            append!(stack,node.args)
        end
        # Do not infer imports from quoted code, macros or function bodies.
        # This deliberately incomplete analyzer is NOT a runtime/type proof.
    end
    (;parse_errors,imports=sort!(unique(string.(imports))),
        outside_published_contract=sort!(unique(string.(filter(x->x∉ALLOWED,imports)))),
        scope="Public syntax/dependency-contract diagnostics only; generated code not executed; no JET, runtime correctness, security or mathematical guarantee")
end

function render(report)
    lines=["Public static tool report (NO benchmark tests or expected answers):"]
    isempty(report.parse_errors) ? push!(lines,"No parse error reported; this does NOT prove runtime or mathematical correctness.") : append!(lines,"Parser: ".*report.parse_errors)
    isempty(report.outside_published_contract) ? push!(lines,"No import outside the published package contract was detected.") :
        push!(lines,"Imports outside the already-published package contract: "*join(report.outside_published_contract,", ")*". Do not install packages or assume these are available.")
    join(lines,"\n")
end
end
