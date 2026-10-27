module PublicJuliaSignatures

# Input types come ONLY from public docstring declarations, not gold bodies or
# test values. Parsing builds text for the isolated JET subprocess; never eval.
const BASE_TYPES=Set([:Any,:Int,:Int64,:Int32,:Integer,:Float64,:Float32,:Real,:Number,
    :Bool,:String,:AbstractString,:Char,:Nothing,:Missing,:Union,:Vector,:AbstractVector,
    :Matrix,:AbstractMatrix,:Array,:AbstractArray,:Tuple,:NTuple,:Dict,:AbstractDict])
function safe_type(expression)
    expression isa Symbol && return expression in BASE_TYPES
    expression isa Integer && return 0<=expression<=16
    expression isa Expr && expression.head==:curly || return false
    all(safe_type,expression.args)
end

function from_declaration(declaration::AbstractString)
    expression=Meta.parse(declaration)
    call=expression isa Expr && expression.head==:(::) ? expression.args[1] : expression
    call isa Expr && call.head==:call && first(call.args) isa Symbol || throw(ArgumentError("function declaration required"))
    name=String(first(call.args))
    occursin(r"^[A-Za-z_][A-Za-z0-9_!]*$",name) || throw(ArgumentError("simple public function name required"))
    types=String[]
    for arg in call.args[2:end]
        if arg isa Symbol
            push!(types,"Any")
        else
            arg isa Expr && arg.head==:(::) && length(arg.args)==2 && arg.args[1] isa Symbol ||
                throw(ArgumentError("unsupported public argument declaration"))
            safe_type(arg.args[2]) || throw(ArgumentError("unsupported public type expression"))
            push!(types,sprint(Base.show_unquoted,arg.args[2]))
        end
    end
    (;name,types="[Tuple{"*join(types,",")*"}]")
end

function from_task_prompt(prompt)
    pieces=split(prompt,"\n\n";limit=2)
    length(pieces)==2 || throw(ArgumentError("missing public docstring"))
    lines=filter(!isempty,strip.(split(last(pieces),'\n')))
    isempty(lines) && throw(ArgumentError("empty public docstring"))
    from_declaration(first(lines))
end
end
