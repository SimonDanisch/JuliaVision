# OIDN HIP folds bias, activation and fp16 write-back into the matrix kernel.
# Keep that epilogue in registers rather than storing a whole fp32 intermediate.
const DK = DNNKernels
struct ConvStore{B, A}
    bias::B
    p0::Int32
    npqc::Int32
    npq::Int32
    cout::Int32
end
function Mantle.Adapt.adapt_structure(to, e::ConvStore{B, A}) where {B, A}
    bias = Mantle.Adapt.adapt(to, Mantle.storage(e.bias))
    return ConvStore{typeof(bias), A}(bias, e.p0, e.npqc, e.npq, e.cout)
end
@inline function Mantle.accstore!(
        out, off, ld,
        c::KI.AcceleratedMatrix{Float32, 16, 16, KI.Accumulator}, e::ConvStore{B, A}
    ) where {B, A}
    row = Int32((off - 1) % ld); col = Int32((off - 1) ÷ ld)
    # Only whole 16x16 tiles are admitted by the lowering below.
    if row < e.npqc && col < e.cout
        b = KI.AcceleratedMatrix{eltype(e.bias), 16, 16, KI.Accumulator}(pointer(e.bias), col + Int32(1), 0, Val(true))
        v = KI.coopmat_add(c, convert(KI.AcceleratedMatrix{Float32, 16, 16, KI.Accumulator}, b))
        if A === :relu
            for i in Int32(0):(KI.coopmat_length(KI.AcceleratedMatrix{Float32, 16, 16, KI.Accumulator}) - Int32(1))
                v = KI.coopmat_setcomp(v, i, max(KI.coopmat_getcomp(v, i), 0.0f0))
            end
        end
        half = convert(KI.AcceleratedMatrix{eltype(out), 16, 16, KI.Accumulator}, v)
        copyto!(pointer(out), Int32(1) + e.p0 + row + col * e.npq, e.npq, half)
    end
    return nothing
end
function fusedgraph(g)
    ops = [op.aten == "convolution.default" ? DK.Op(op.id, "oidn.convolution", op.ins, op.out, op.attrs) : op for op in g.ops]
    return DK.Graph(g.name, g.symbols, g.inputs, g.outputs, g.buffers, g.order, ops, g.fusion)
end
function DK.emitop!(ctx::DK.EmitCtx, op::DK.Op, ::Val{Symbol("oidn.convolution")})
    x = DK.operand(ctx, op.ins[1]); w = DK.operand(ctx, op.ins[2])
    bias = length(op.ins) >= 3 ? DK.operand(ctx, op.ins[3]) : nothing
    out = DK.dest(ctx)
    fallback() = DK.emitop!(ctx, op, Val(Symbol("convolution.default")))
    eltype(out) === Float16 && bias !== nothing && size(x, 4) == 1 || return fallback()
    stride = reverse(DK.ints(op.attrs["arg3"])); pad = reverse(DK.ints(op.attrs["arg4"]))
    dil = reverse(DK.ints(op.attrs["arg5"]))
    DK.atenarg(op, 6, "transposed", false) == false && DK.atenarg(op, 8, "groups", 1) == 1 || return fallback()
    act = Symbol(get(op.attrs, "act", :none))
    act in (:none, :relu) && size(out, 3) % 16 == 0 || return fallback()
    plan = DK.conv_coopmat_plan(DK.caps(ctx.dev), eltype(x), eltype(w), size(out), size(w))
    plan isa DK.ConvCoopMatPlan || return fallback()
    MP = DK.planmp(plan)
    blk_split = Mantle.coopmat_gemm_shape(MP, plan.CoutP, plan.CRSP)
    blk_split[2] == 1 || return fallback()
    # A partial last tile cannot be stored by a cooperative matrix operation.
    plan.rows % 16 == 0 && plan.NPQ % 16 == 0 || return fallback()
    KW, KH, Cin, Cout = size(w); W, H = size(x, 1), size(x, 2)
    OW, OH = size(out, 1), size(out, 2)
    gathering = plan.gathertiling !== nothing
    col = gathering ? nothing : DK.scratch(ctx, Float16, MP, plan.CRSP)
    B = if plan.CRSP == plan.CRS && plan.CoutP == Cout
        Mantle.viewof(w, (plan.CRS, Cout))
    else
        wp = DK.scratch(ctx, Float16, plan.CRSP, plan.CoutP)
        Mantle.dispatch!(ctx.g, Mantle.fill_kernel!, (wp, zero(Float16)), length(wp); name = "$(op.id).wzero")
        DK.blockcopydispatch!(
            ctx, wp, (plan.CRSP, plan.CoutP), Mantle.viewof(w, (plan.CRS, Cout)),
            (plan.CRS, Cout), (0, 0); name = "$(op.id).wpad"
        )
        wp
    end
    for chunk in 0:(cld(plan.NPQ, plan.rows) - 1)
        p0 = chunk * plan.rows; npqc = min(plan.rows, plan.NPQ - p0)
        epi = ConvStore{typeof(bias), act}(bias, Int32(p0), Int32(npqc), Int32(plan.NPQ), Int32(Cout))
        if gathering
            Mantle.coopmat_gemm_dispatch!(
                ctx.g, out, x, B, MP, plan.CoutP, plan.CRSP;
                blk_split, reduce = false, tiling = plan.gathertiling, epilogue = epi,
                aload = DK.ConvGather(OW, OH, KW, KH, stride, pad, dil, W, H, Cin, npqc, p0),
                name = "$(op.id).fused.$chunk"
            )
        else
            vec = MP % DK.IM2COL_VEC == 0 ? DK.IM2COL_VEC : 1
            Mantle.dispatch!(
                ctx.g, DK.im2col_kernel!,
                (
                    col, x, Val(MP), Val(vec), Val(KW), Val(KH), Val(stride[1]), Val(stride[2]),
                    Val(pad[1]), Val(pad[2]), Val(dil[1]), Val(dil[2]), Val(OW), Val(OH),
                    W, H, npqc, MP * plan.CRSP, Cin, p0,
                ), cld(MP * plan.CRSP, vec);
                name = "$(op.id).im2col.$chunk"
            )
            Mantle.coopmat_gemm_dispatch!(
                ctx.g, out, col, B, MP, plan.CoutP, plan.CRSP;
                blk_split, reduce = false, epilogue = epi, name = "$(op.id).fused.$chunk"
            )
        end
    end
    return out
end
