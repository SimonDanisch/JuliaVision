"""
Reference images: every host-side step against the reference's own code, the
new graphs against PyTorch, and the segmented denoiser against the reference's
block-causal forward.

The fixtures come from `tools/export_qwenimage21_condition.py`:

* `condition_host` runs two synthetic images through Pillow's resize and white
  composite, `Qwen3VLProcessor`, `get_rope_index`, the vision tower's grid
  helpers and `QwenImage21Rope`. Each of those is bookkeeping the runner ports,
  and each gets a picture subtly wrong rather than failing when it is off by a
  row: a transposed patch order or a frame axis that does not freeze over an
  image still produces an image.
* `condition_vision` and `condition_vae_encoder` are the vision tower and the
  VAE encoder at Float32 with the real weights. The vision tower is what found
  `fused.rope` reading its tables by head instead of by token (see DNNKernels'
  `test_fuserope_layouts.jl`): 99% wrong, no error.
* `smoke_*` is the four denoiser graphs exported from a random two-block model,
  and that model's full forward over two reference images between text runs.
"""

using Test
using QwenImageRunner
using QwenImageRunner: refsdir, ready, condition_size, pil_resize, over_white, vision_patches,
    vision_grid_inputs, condition_tokens, mrope_positions, joint_layout, IMAGE_PAD_ID
using DNNKernels
using Mantle

relerr(a, b) = sqrt(sum(abs2, Float32.(a) .- Float32.(b)) / sum(abs2, Float32.(b)))

@testset "reference images: the host side is the reference's" begin
    r = DNNKernels.readsafetensors(joinpath(refsdir(), "condition_host.safetensors"))
    resolution = 320
    sizes = [Tuple(r["sizes"][:, k]) for k in 1:size(r["sizes"], 2)]
    grids = Tuple{Int,Int}[]
    for k in 0:1
        orig = r["orig_$k"]
        # `calculate_dimensions`, including the half-to-even 12.5 -> 12 of the
        # second image.
        @test condition_size(size(orig, 2), size(orig, 3); resolution) == sizes[k+1]
        # Pillow's LANCZOS on premultiplied RGBA, to the byte.
        resized = pil_resize(orig, sizes[k+1]...)
        @test resized == r["resized_$k"]
        white = over_white(resized)
        @test white == r["white_$k"]
        patches, grid = vision_patches(white)
        push!(grids, grid)
        idx, w, pos = vision_grid_inputs(grid...)
        @test idx == r["interp_idx_$k"]
        @test w ≈ r["interp_w_$k"] atol = 1f-7
        @test pos == r["vispos_$k"]
    end
    @test grids == [(Int(r["grid"][2, k]), Int(r["grid"][3, k])) for k in 1:2]
    # `Qwen2VLImageProcessorFast`'s patches, in its order (merge blocks, then
    # `(C, T, 16, 16)`), stored at Float16.
    pixel = hcat((first(vision_patches(over_white(pil_resize(r["orig_$k"], sizes[k+1]...))))
                  for k in 0:1)...)
    @test maximum(abs.(pixel .- Float32.(r["pixel_values"]))) <= 1f-3

    tk = QwenTokenizer(QwenImageRunner.processordir())
    prompt = "Put the object from the first picture onto the second one."
    ids = condition_tokens(tk, prompt, grids)
    @test ids == r["input_ids"]
    @test mrope_positions(ids, grids) == permutedims(r["mrope"])

    # The denoiser's joint sequence: the system turn dropped, the slots widened
    # to four latent tokens, the target appended, and `QwenImage21Rope`'s
    # positions over all of it.
    slots = ids[15:end] .== IMAGE_PAD_ID
    @test slots == (r["img_mask"] .== 0x01)
    shapes = [(s[2] ÷ 16, s[1] ÷ 16) for s in sizes]
    layout = joint_layout(slots, vcat(shapes, [last(shapes)]))
    @test maximum(abs.(layout.cos .- r["rope_real"])) <= 1f-6
    @test maximum(abs.(layout.sin .- r["rope_imag"])) <= 1f-6
    @test [first(s) for s in layout.segments] == [:text, :image, :text, :image, :text]
    @test [length(last(s)) for s in layout.segments if first(s) === :image] == [h * w for (h, w) in shapes]

    # A small `resolution` can round an image under the processor's pixel floor,
    # where the reference pipeline itself raises; here it is refused up front.
    small = pil_resize(r["orig_0"], condition_size(300, 190; resolution = 256)...)
    @test_throws ArgumentError vision_patches(over_white(small))
    # Positions for more images than the prompt has slots for.
    one = condition_tokens(tk, prompt, grids[1:1])
    @test_throws ArgumentError mrope_positions(one, grids)
end

@testset "reference images: the segmented denoiser is the reference forward" begin
    backend = Mantle.defaultbackend()
    dir = refsdir()
    names = (QwenImageRunner.PREFIXGRAPHS..., QwenImageRunner.STEPGRAPH)
    graphs = Dict(n => DNNKernels.loadgraph(joinpath(dir, "smoke_$n.json")) for n in names)
    weights = Dict{String,Any}("model." * k => v
                               for (k, v) in DNNKernels.readsafetensors(joinpath(dir, "smoke_weights.safetensors")))
    for n in names
        merge!(weights, DNNKernels.readsafetensors(joinpath(dir, "smoke_$(n)_constants.safetensors")))
    end
    t = QwenImageRunner.QwenConditionedTransformer(backend, DNNKernels.Model(graphs, weights; backend))
    try
        ref = DNNKernels.readsafetensors(joinpath(dir, "smoke_reference.safetensors"))
        vl = vec(ref["img_mask"])
        shapes = [(Int(ref["img_shapes"][2, k]), Int(ref["img_shapes"][3, k])) for k in 1:3]
        # The target's slots are the pipeline's addition, not the prompt's.
        prompt_slots = vl[1:end - prod(last(shapes)) ÷ 4] .== 0x01
        # The smoke model's rotary axes, `axes_dims_rope=(4, 6, 6)`.
        layout = joint_layout(prompt_slots, shapes; axes = (4, 6, 6))
        @test layout.cos ≈ ref["rotary_real"] atol = 1f-6
        @test layout.sin ≈ ref["rotary_imag"] atol = 1f-6
        embeds = ref["prompt"][:, :, 1]
        latents = ref["latents"][:, :, 1]
        n1, n2 = prod(shapes[1]), prod(shapes[2])
        refs = [latents[:, 1:n1], latents[:, n1+1:n1+n2]]
        cache = QwenImageRunner.prefill(t, layout, embeds, refs)
        @test length(cache) == 2 * 2
        @test all(size(c, 3) == layout.prefix for c in cache)
        dev(x) = DNNKernels.toback(backend, x)
        target = layout.prefix+1:layout.total
        got = DNNKernels.call(t.model, QwenImageRunner.STEPGRAPH,
                              dev(reshape(Float16.(latents[:, n1+n2+1:end]), size(latents, 1), :, 1)),
                              dev(Float16.(ref["timestep"])), dev(layout.cos[:, target]),
                              dev(layout.sin[:, target]), (dev(c) for c in cache)...;
                              dims = (i = length(target), p = layout.prefix))
        # Float16 graphs against the Float32 forward on the same weights.
        @test relerr(Array(first(got)), ref["noise"]) < 5f-3
    finally
        Mantle.release!(t)
    end
end

@testset "reference images: vision tower and VAE encoder against PyTorch" begin
    if ready(:condition_encoder) && ready(:vae_encoder)
        backend = Mantle.defaultbackend()
        dev(x) = DNNKernels.toback(backend, x)
        enc = qwenimageconditionencoder(; backend)
        try
            v = DNNKernels.readsafetensors(joinpath(refsdir(), "condition_vision.safetensors"))
            gh, gw = v["grid"]
            outs = DNNKernels.call(enc.model, QwenImageRunner.VISIONGRAPH, dev(v["patches"]),
                                   dev(v["interp_indices"]), dev(Float32.(v["interp_weights"])),
                                   dev(v["position_ids"]); dims = (n = gh * gw ÷ 4,))
            # Float16 attention inside a Float32 tower, against Float32.
            @test relerr(Array(outs[1]), v["merged"]) < 5f-3
            for d in 0:2
                @test relerr(Array(outs[d+2]), v["deepstack_$d"]) < 5f-3
            end
        finally
            Mantle.release!(enc)
        end
        vae = qwenimagevaeencoder(; backend)
        try
            e = DNNKernels.readsafetensors(joinpath(refsdir(), "condition_vae_encoder.safetensors"))
            img = e["image"]
            lat = DNNKernels.call(vae.model, QwenImageRunner.VAEENCODERGRAPH, dev(img);
                                  dims = (h = size(img, 2), w = size(img, 1)))
            @test relerr(Array(first(lat)), e["latents"]) < 1f-5
        finally
            Mantle.release!(vae)
        end
    else
        @test_skip false
    end
end
