"""
Render the model demos as a Bonito app, built from DOM nodes.

    include("tools/demopage.jl"); demopage()

Deliberately *not* the generated `index.html`. That file is a standalone
document — doctype, `<style>` with `body { ... }` — and handing it to `HTML()`
puts those global rules into the page the chat itself lives in, which restyles
everything around it. An app rendered into a shared page has to build elements
and style them locally, which is what this does: every rule is an inline `style`
on a node this function created.

`index.html` stays as the artefact to open from disk; this is the in-chat view.
"""

# JSON through Bonito rather than as a direct dependency: the manifests are four
# fields each, and adding a package to the editor's Project.toml to read them
# would be a resolve for nothing.
using Bonito
using Bonito: JSON

const DEMODIR = joinpath(@__DIR__, "..", "media", "model-demos")
const ORDER = ["whisper", "neurallut", "rife", "deepfilternet", "depthanything",
               "basicvsrpp", "demucs", "kokoro", "propainter", "fluxklein"]

const CARD = "background:#1a1d23;border-radius:6px;padding:1rem 1.2rem;margin:1.2rem 0"
const CAP = "color:#9aa;font-size:.86rem;margin:.3rem 0 1rem"
const MEDIA = "max-width:100%;border-radius:5px;display:block;background:#000"

"""One artefact: a video, an audio player, or an image, plus its caption."""
function artefact(session, dir, model, art)
    file = art["file"]
    path = normpath(joinpath(dir, model, file))
    isfile(path) || return DOM.div("missing: $file"; style = CAP)
    url = Bonito.url(session, Bonito.Asset(path))
    ext = lowercase(splitext(file)[2])
    node = if ext == ".mp4"
        DOM.video(; src = url, controls = true, loop = true,
                  preload = "metadata", style = MEDIA)
    elseif ext in (".wav", ".mp3")
        DOM.audio(; src = url, controls = true, preload = "none",
                  style = "width:100%;margin-top:.2rem")
    else
        DOM.img(; src = url, loading = "lazy", style = MEDIA)
    end
    return DOM.figure(node, DOM.figcaption(art["caption"]; style = CAP);
                      style = "margin:1rem 0")
end

"""One model: heading, what to judge, the measured numbers, then the artefacts."""
function section(session, dir, man, i)
    kids = Any[DOM.h2("$i. " * man["title"];
                      style = "font-size:1.2rem;color:#fff;margin:.2rem 0 .6rem")]
    hint = get(man, "verdict_hint", "")
    isempty(hint) || push!(kids, DOM.div(DOM.b("What to judge: "), hint;
        style = "background:#1c2028;border-left:3px solid #5a8;padding:.6rem .9rem;" *
                "border-radius:0 4px 4px 0;margin-bottom:.6rem"))
    notes = get(man, "notes", String[])
    isempty(notes) || push!(kids, DOM.div(
        [DOM.code(n; style = "background:#22262e;padding:.1rem .35rem;border-radius:3px;" *
                             "margin-right:.5rem;display:inline-block") for n in notes]...;
        style = CAP))
    tr = joinpath(dir, man["model"], "transcripts.md")
    isfile(tr) && push!(kids, DOM.pre(read(tr, String);
        style = "background:#14161a;padding:.8rem;border-radius:4px;white-space:pre-wrap;" *
                "font-size:.85rem;color:#cde"))
    for art in man["artifacts"]
        push!(kids, artefact(session, dir, man["model"], art))
    end
    return DOM.section(kids...; style = CARD)
end

function demopage(dir = DEMODIR)
    App() do session
        mans = [JSON.parsefile(joinpath(dir, m, "manifest.json"))
                for m in ORDER if isfile(joinpath(dir, m, "manifest.json"))]
        DOM.div(
            DOM.h1("Candidate models — what they actually do";
                   style = "font-size:1.5rem;color:#fff;margin:0 0 .3rem"),
            DOM.div("PyTorch on this machine (RTX 4000 Ada, 20 GB), on real footage " *
                    "from this project. Timings are the target to beat, not a port result.";
                    style = CAP),
            [section(session, dir, m, i) for (i, m) in enumerate(mans)]...;
            style = "font:15px/1.6 -apple-system,Segoe UI,Roboto,sans-serif;" *
                    "color:#e6e6e6;max-width:1000px")
    end
end
