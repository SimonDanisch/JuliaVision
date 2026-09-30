# The JuliaVision documentation: every package README as one static site, built
# with Bonito.
#
#     julia --project=<an environment with Bonito> docs/make.jl
#
# writes `docs/build/`, which any static file server can host. Nothing here runs
# a model: the pictures are the ones `docs/examples/*.jl` wrote into `media/`.
using Bonito
using Bonito: DOM, DontEscape

const REPO = normpath(joinpath(@__DIR__, ".."))

"One page of the site. `route` is `\"\"` for the index; every other page is one directory deep."
struct Page
    route::String
    title::String
    section::String
    readme::String      # repository-relative path of the markdown, `""` when generated
    markdown::String
end

readmepage(pkg, section) =
    Page(lowercase(pkg), pkg, section, "$pkg/README.md", read(joinpath(REPO, pkg, "README.md"), String))

"The example scripts, each as a section: the code behind every picture on the site."
function examplespage()
    io = IOBuffer()
    println(io, "# Example code\n")
    println(io, "Every picture and sound on this site was made by one of these scripts. ",
            "Each defines a function, and running the file calls it and rewrites its part of `media/`. ",
            "`common.jl` holds the image helpers they share.\n")
    for file in sort(readdir(joinpath(REPO, "docs", "examples")))
        endswith(file, ".jl") || continue
        println(io, "## ", file, "\n")
        println(io, "```julia")
        print(io, read(joinpath(REPO, "docs", "examples", file), String))
        println(io, "```\n")
    end
    return Page("examples", "Example code", "Code", "", String(take!(io)))
end

function sitepages()
    sections = ["Runtime" => ["DNNKernels", "GPUFiltering"],
                "Images" => ["QwenImageRunner", "SAM2Runner", "MatAnyoneRunner", "DepthAnythingRunner",
                             "NeuralLUTRunner", "RIFERunner", "BasicVSRRunner"],
                "3D" => ["Hunyuan3DRunner", "Trellis2Runner"],
                "Audio" => ["WhisperRunner", "KokoroRunner"],
                "Language" => ["BonsaiRunner", "HorizonRunner"]]
    pages = [Page("", "Overview", "JuliaVision", "README.md", read(joinpath(REPO, "README.md"), String))]
    for (section, pkgs) in sections, pkg in pkgs
        push!(pages, readmepage(pkg, section))
    end
    push!(pages, examplespage())
    return pages
end

"Where `dest`, written in `page`'s markdown, points on the site."
function sitelink(dest::AbstractString, page::Page, pages::Vector{Page}, sourceurl::AbstractString)
    occursin(r"^(https?:|mailto:|#)", dest) && return dest
    up = isempty(page.route) ? "" : "../"
    path, anchor = let i = findfirst('#', dest)
        i === nothing ? (dest, "") : (dest[1:i-1], dest[i:end])
    end
    p = normpath(joinpath(dirname(page.readme), path))
    startswith(p, "media/") && return dest
    target = findfirst(q -> q.readme == p, pages)
    target !== nothing && return up * (isempty(pages[target].route) ? "" : pages[target].route * "/") * anchor
    if startswith(p, "docs/examples")
        name = basename(rstrip(p, '/'))
        return up * "examples/" * (endswith(name, ".jl") ? "#" * name : "")
    end
    return sourceurl * rstrip(p, '/')
end

"The markdown with its links pointing at the site, and MP3 links as players."
function relink(page::Page, pages::Vector{Page}, sourceurl::AbstractString)
    md = replace(page.markdown, r"\[([^\]]*)\]\(([^)\s]+\.mp3)\)" =>
                 s"<audio controls preload=\"none\" src=\"\2\"></audio>")
    return replace(md, r"\]\(([^)\s]+)\)" => m -> begin
        dest = match(r"\]\(([^)\s]+)\)", m).captures[1]
        "](" * sitelink(dest, page, pages, sourceurl) * ")"
    end)
end

function sidebar(page::Page, pages::Vector{Page})
    up = isempty(page.route) ? "" : "../"
    items = Any[]
    section = ""
    for q in pages
        if q.section != section
            section = q.section
            push!(items, DOM.div(section; class = "section"))
        end
        href = up * (isempty(q.route) ? "" : q.route * "/")
        push!(items, DOM.a(q.title; href, class = q.route == page.route ? "current" : ""))
    end
    return DOM.nav(items...; class = "sidebar")
end

# highlight.js from a CDN: code stays readable, just plain, where it cannot load.
const HIGHLIGHT = """
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/julia.min.js"></script>
<script>hljs.highlightAll();</script>
"""

const CSS = """
body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
       color: #1f2328; background: #ffffff; line-height: 1.55; }
.sidebar { position: fixed; top: 0; left: 0; bottom: 0; width: 230px; overflow-y: auto;
           padding: 20px 16px; background: #f6f8fa; border-right: 1px solid #d0d7de; box-sizing: border-box; }
.sidebar .section { margin: 16px 0 4px; font-size: 12px; font-weight: 600; text-transform: uppercase;
                    letter-spacing: 0.05em; color: #59636e; }
.sidebar a { display: block; padding: 3px 8px; border-radius: 6px; color: #1f2328; text-decoration: none; font-size: 14px; }
.sidebar a:hover { background: #eaeef2; }
.sidebar a.current { background: #ddf4ff; color: #0969da; font-weight: 600; }
main { margin-left: 230px; padding: 24px 48px 80px; }
.markdown-body { max-width: 1000px; }
.markdown-body h1 { border-bottom: 1px solid #d0d7de; padding-bottom: 0.3em; }
.markdown-body h2 { border-bottom: 1px solid #d0d7de; padding-bottom: 0.3em; margin-top: 1.8em; }
.markdown-body img { max-width: 100%; height: auto; }
.markdown-body table { border-collapse: collapse; margin: 1em 0; display: block; overflow-x: auto; }
.markdown-body th, .markdown-body td { border: 1px solid #d0d7de; padding: 6px 12px; vertical-align: top; }
.markdown-body tr:nth-child(2n) { background: #f6f8fa; }
.markdown-body pre { background: #f6f8fa; border-radius: 6px; padding: 12px 16px; overflow-x: auto; font-size: 13.5px; }
.markdown-body pre code.hljs { background: transparent; padding: 0; }
.markdown-body code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.9em; }
.markdown-body :not(pre) > code { background: #eff1f3; border-radius: 4px; padding: 0.15em 0.35em; }
.markdown-body blockquote { margin: 1em 0; padding: 0 1em; color: #59636e; border-left: 4px solid #d0d7de; }
.markdown-body a { color: #0969da; text-decoration: none; }
.markdown-body a:hover { text-decoration: underline; }
.markdown-body audio { height: 32px; vertical-align: middle; }
@media (max-width: 800px) { .sidebar { position: static; width: auto; } main { margin-left: 0; padding: 16px; } }
"""

function pageapp(page::Page, pages::Vector{Page}, sourceurl::AbstractString)
    body = Bonito.commonmark_to_dom(Bonito.bonito_parser(relink(page, pages, sourceurl)))
    return App(DOM.div(DOM.style(CSS), sidebar(page, pages), DOM.main(body), DontEscape(HIGHLIGHT));
               title = page.route == "" ? "JuliaVision" : "JuliaVision: $(page.title)")
end

function makedocs(; build = joinpath(REPO, "docs", "build"))
    branch = readchomp(`git -C $REPO rev-parse --abbrev-ref HEAD`)
    sourceurl = "https://github.com/SimonDanisch/JuliaVision/blob/$branch/"
    pages = sitepages()
    isdir(build) && rm(build; recursive = true)
    routes = Routes()
    for page in pages
        routes["/" * page.route] = pageapp(page, pages, sourceurl)
    end
    export_static(build, routes)
    cp(joinpath(REPO, "media"), joinpath(build, "media"))
    return build
end

abspath(PROGRAM_FILE) == (@__FILE__) && makedocs()
