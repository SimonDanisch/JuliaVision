# Packed products are DNN operations, not properties of one model runner.
# These overloads are available without loading BonsaiRunner, and retain the
# existing capability-selected PTQ1 arithmetic when the IR is lowered.
function ptq1_rows!(g::SemanticGraph,out,A::PTQ1Matrix,x;name::AbstractString)
    eltype(x)===Float32 || throw(ArgumentError("PTQ1 decode rows require Float32 activations"))
    semanticop!(g,"dnn.ptq1_rows",(out,A.data,x);
        attrs=Dict{String,Any}("m"=>A.m,"k"=>A.k,"label"=>String(name)))
    nothing
end
function ptq1_rows2!(g::SemanticGraph,out1,A1::PTQ1Matrix,out2,A2::PTQ1Matrix,x;name::AbstractString)
    eltype(x)===Float32 || throw(ArgumentError("PTQ1 decode rows require Float32 activations"))
    A1.k==A2.k || throw(DimensionMismatch("Packed pair widths differ"))
    semanticop!(g,"dnn.ptq1_rows_pair",(out1,A1.data,out2,A2.data,x);writes=(1,3),
        attrs=Dict{String,Any}("m1"=>A1.m,"m2"=>A2.m,"k"=>A1.k,"label"=>String(name)))
    nothing
end
function ptq1_cols!(g::SemanticGraph,out,A::PTQ1Matrix,x,first::Integer,N::Integer;name::AbstractString)
    semanticop!(g,"dnn.ptq1_columns",(out,A.data,x);
        attrs=Dict{String,Any}("m"=>A.m,"k"=>A.k,"first"=>Int(first),"n"=>Int(N),"label"=>String(name)))
    nothing
end
function ptq1_native_gemm!(g::SemanticGraph,out,A::PTQ1Matrix,x,N::Integer;name::AbstractString)
    cover=32*(Int(N)÷32)
    eltype(x)===Float16 && cover>0 || return 0
    M.native_gemm_available(g.dev,Float16,Float16,eltype(out)) || return 0
    semanticop!(g,"dnn.ptq1_native_product",(out,A.data,x);
        attrs=Dict{String,Any}("m"=>A.m,"k"=>A.k,"n"=>Int(N),"cover"=>cover,"label"=>String(name)))
    cover
end
for kind in ("rows","rows_pair","columns","native_product")
    tag=Symbol("dnn.ptq1_$kind")
    @eval function emitop!(ctx::EmitCtx,op::Op,::Val{$(QuoteNode(tag))})
        emit_semantic_ptq!(ctx,op,Val($(QuoteNode(Symbol(kind)))))
    end
end
function emit_semantic_ptq!(ctx,op,::Val{:rows})
    out,data,x=semanticargument(ctx,op,op.attrs["arguments"]);a=op.attrs
    ptq1_rows!(ctx.g,out,PTQ1Matrix(data,a["m"],a["k"]),x;name=a["label"])
end
function emit_semantic_ptq!(ctx,op,::Val{:rows_pair})
    out1,data1,out2,data2,x=semanticargument(ctx,op,op.attrs["arguments"]);a=op.attrs
    ptq1_rows2!(ctx.g,out1,PTQ1Matrix(data1,a["m1"],a["k"]),
        out2,PTQ1Matrix(data2,a["m2"],a["k"]),x;name=a["label"])
end
function emit_semantic_ptq!(ctx,op,::Val{:columns})
    out,data,x=semanticargument(ctx,op,op.attrs["arguments"]);a=op.attrs
    ptq1_cols!(ctx.g,out,PTQ1Matrix(data,a["m"],a["k"]),x,a["first"],a["n"];name=a["label"])
end
function emit_semantic_ptq!(ctx,op,::Val{:native_product})
    out,data,x=semanticargument(ctx,op,op.attrs["arguments"]);a=op.attrs
    cover=ptq1_native_gemm!(ctx.g,out,PTQ1Matrix(data,a["m"],a["k"]),x,a["n"];name=a["label"])
    cover==a["cover"] || error("Packed native product capability changed during lowering")
end
