using Test, PPOCRRunner, DNNKernels, ColorTypes
import PNGFiles
using PPOCRRunner: detplan, oriplan, recplan, linewidth, REC_WIDTHS

# What the exporter leaves beside the graphs: PyTorch's answers on the inputs it
# exported with. Not in the artifacts (the people using the runner do not need
# them), so these checks run where the exports were made and are skipped
# elsewhere.
const GEN = let dir = normpath(joinpath(@__DIR__, "..", "..", "gen", "graphs"))
    isdir(dir) ? dir : normpath(joinpath(@__DIR__, "..", "..", "..", "..", "gen", "graphs"))
end
reference(name) = (p = joinpath(GEN, name, "reference.safetensors"); isfile(p) ? readsafetensors(p) : nothing)

@testset "PPOCRRunner" begin
    if !PPOCRRunner.ready()
        @info "PP-OCRv6 exports not installed; nothing to test"
        return
    end
    m = ppocr()
    try
        @testset "against PyTorch" begin
            ref = reference("ppocrv6_det")
            if ref !== nothing
                input = only(detplan(m).inputs)
                copyto!(input, ref["input"])
                out = Array(first(DNNKernels.call(m.det, "ppocrv6_det", input; dims = (;))))
                @test maximum(abs.(out .- ref["probability"])) < 5e-3
            end
            ref = reference("ppocr_docori")
            if ref !== nothing
                input = only(oriplan(m).inputs)
                copyto!(input, ref["input"])
                out = vec(Array(first(DNNKernels.call(m.ori, "ppocr_docori", input; dims = (;)))))
                @test maximum(abs.(out .- vec(ref["logits"]))) < 1e-3
            end
            ref = reference("ppocrv6_rec")
            if ref !== nothing
                x = ref["input"]
                W = size(x, 1)
                input = only(recplan(m, W).inputs)
                copyto!(input, x)
                best, index = DNNKernels.call(m.rec, "ppocrv6_rec", input; dims = (w = W ÷ 16,))
                # The index per time step, not just the best value: `max.dim` once
                # returned the right maxima at the wrong positions.
                @test Array(index) == ref["index"]
            end
        end

        @testset "an empty page" begin
            page = fill(RGB{Float32}(0.95f0, 0.95f0, 0.95f0), 620, 877)
            @test isempty(detect(m, page))
            @test isempty(read_lines(m, page))
            @test orientation(m, page) in (0, 90, 180, 270)
        end

        # A crop of a synthetic letter, 300 dpi, eight lines.
        letter = permutedims(PNGFiles.load(joinpath(@__DIR__, "letter.png")))
        expected = ["Sehr geehrte Damen und Herren,",
            "hiermit bestätigen wir den Eingang Ihrer Unterlagen vom 12. März 2026.",
            "Die Prüfung Ihres Antrags auf Erstattung der Gebühren ist abgeschlossen.",
            "Der Betrag von 1.284,50 € wird innerhalb von vierzehn Tagen überwiesen.",
            "Bitte beachten Sie, dass Änderungen Ihrer Bankverbindung schriftlich",
            "mitgeteilt werden müssen. Für Rückfragen stehen wir Ihnen gerne zur Verfügung.",
            "Unsere Öffnungszeiten sind Montag bis Freitag von 8:00 bis 16:30 Uhr.",
            "Größere Beträge über 5.000 € erfordern eine zusätzliche Bestätigung."]

        @testset "a letter" begin
            lines = read_lines(m, letter)
            @test [l.text for l in lines] == expected
            @test all(l -> l.score > 0.9, lines)
            # Words carry their extent along the line, inside the line's box.
            first_line = first(lines)
            @test first.(first_line.words) == split(expected[1])
            @test all(w -> first_line.box[1] <= w[2] < w[3] <= first_line.box[3], first_line.words)
        end

        @testset "every page type is converted on the device" begin
            boxes = detect(m, letter)
            @test detect(m, RGB{Float32}.(letter)) == boxes
            @test detect(m, RGB{eltype(eltype(letter))}.(letter)) == boxes
            # One page input per size, reused by the next upload.
            @test pageinput!(m, size(letter)) === upload!(m, letter)
        end

        @testset "which way up" begin
            @test orientation(m, letter) == 0
            @test orientation(m, permutedims(letter)[:, end:-1:1]) == 90
            @test orientation(m, letter[end:-1:1, end:-1:1]) == 180
            @test orientation(m, permutedims(letter)[end:-1:1, :]) == 270
        end

        @testset "a curved line is read again, wider" begin
            # The footer of a bent invoice: its box is tall for its letters, so
            # at 48 pixels high they come out too narrow to read.
            line = permutedims(PNGFiles.load(joinpath(@__DIR__, "curved-line.png")))
            expected = "Diese Rechnung wurde maschinell erstellt und ist ohne Unterschrift gültig"
            page = upload!(m, line)
            box = (0.0f0, 0.0f0, Float32(size(line, 1)), Float32(size(line, 2)))
            narrow = recognize(m, page, box; retry = false)
            @test narrow.score < 0.8
            wide = recognize(m, page, box)
            @test wide.text == expected
            @test wide.score > 0.95
            @test join(first.(wide.words), " ") == expected
            @test all(0 <= a <= b <= size(line, 1) for (_, a, b) in wide.words)
            # The wider reading leaves nothing behind in the narrow plan.
            @test recognize(m, page, box; retry = false).text == narrow.text
        end

        @testset "line widths are the planned ones" begin
            for (h, w) in ((40.0f0, 100.0f0), (40.0f0, 1500.0f0), (20.0f0, 4000.0f0))
                W, rw = linewidth((0.0f0, 0.0f0, w, h))
                @test W in REC_WIDTHS
                @test rw <= W
            end
        end
    finally
        close(m)
    end
end
