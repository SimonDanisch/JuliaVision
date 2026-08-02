"""Collect the demo manifests into one page you can look at and listen to.

    uv run tools/demo_page.py

Writes `media/model-demos/index.md` (portable, but a browser is what plays
video) and `media/model-demos/index.html` (what to actually open — inline
`<video controls>` and `<audio controls>` for every artefact).

Ordered by the port order in `models-to-port.md`, not alphabetically, so the
page reads as the argument it is meant to settle: is this one worth the work.
"""

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "media" / "model-demos"

ORDER = ["whisper", "neurallut", "rife", "deepfilternet", "depthanything",
         "basicvsrpp", "demucs", "kokoro", "propainter", "fluxklein"]

CSS = """
:root { color-scheme: dark; }
body { background:#14161a; color:#e6e6e6; font:15px/1.6 -apple-system,Segoe UI,Roboto,sans-serif;
       max-width:1100px; margin:0 auto; padding:2rem 1.5rem 6rem; }
h1 { font-size:1.7rem; border-bottom:1px solid #333; padding-bottom:.6rem; }
h2 { font-size:1.25rem; margin:2.6rem 0 .2rem; color:#fff; }
h2 .n { color:#666; font-weight:400; margin-right:.5rem; }
.hint { background:#1c2028; border-left:3px solid #5a8; padding:.7rem 1rem; margin:.8rem 0;
        border-radius:0 4px 4px 0; }
.notes { color:#9aa; font-size:.88rem; margin:.4rem 0 1rem; }
.notes code { background:#22262e; padding:.1rem .35rem; border-radius:3px; }
figure { margin:1.1rem 0; }
figcaption { color:#9aa; font-size:.86rem; margin-top:.35rem; }
video, img { max-width:100%; border-radius:5px; display:block; background:#000; }
audio { width:100%; margin-top:.2rem; }
.transcript { background:#1a1d23; border-radius:5px; padding:.2rem 1rem; margin:1rem 0; }
.transcript blockquote { border-left:3px solid #567; margin:.4rem 0; padding-left:.9rem; color:#cde; }
nav a { color:#7ac; margin-right:1rem; font-size:.9rem; }
"""


def media_tag(model, art):
    f, cap = art["file"], html.escape(art["caption"])
    src = f"{model}/{f}"
    ext = Path(f).suffix.lower()
    if ext == ".mp4":
        el = f'<video controls loop preload="metadata" src="{src}"></video>'
    elif ext in (".wav", ".mp3", ".flac"):
        el = f'<audio controls preload="none" src="{src}"></audio>'
    elif ext in (".png", ".jpg", ".jpeg", ".gif"):
        el = f'<img loading="lazy" src="{src}">'
    else:
        el = f'<a href="{src}">{html.escape(f)}</a>'
    return f"<figure>{el}<figcaption>{cap}</figcaption></figure>"


def main():
    mans = []
    for name in ORDER:
        p = OUT / name / "manifest.json"
        if p.exists():
            mans.append(json.loads(p.read_text()))
    missing = [n for n in ORDER if not (OUT / n / "manifest.json").exists()]

    parts = ["<!doctype html><meta charset=utf-8><title>model demos</title>",
             f"<style>{CSS}</style>",
             "<h1>Candidate models — what they actually do</h1>",
             "<p class=notes>Every clip below is PyTorch on this machine "
             "(RTX 4000 Ada, 20 GB), on real footage from this project where the "
             "model takes video. Timings are PyTorch, so they are the target to "
             "beat, not the result of a port. Order is the port order from "
             "<code>models-to-port.md</code>.</p>",
             "<nav>" + " ".join(
                 f'<a href="#{m["model"]}">{m["model"]}</a>' for m in mans) + "</nav>"]
    md = ["# Candidate models — what they actually do", "",
          "PyTorch on this machine, real footage. Timings are the target to beat.",
          "**Open `index.html` for the video and audio players.**", ""]

    for i, m in enumerate(mans, 1):
        parts.append(f'<h2 id="{m["model"]}"><span class=n>{i}</span>{html.escape(m["title"])}</h2>')
        if m.get("verdict_hint"):
            parts.append(f'<div class=hint><b>What to judge:</b> {html.escape(m["verdict_hint"])}</div>')
        if m.get("notes"):
            parts.append("<div class=notes>" +
                         " &middot; ".join(f"<code>{html.escape(n)}</code>" for n in m["notes"]) +
                         "</div>")
        md += [f"## {i}. {m['title']}", ""]
        if m.get("verdict_hint"):
            md += [f"**What to judge:** {m['verdict_hint']}", ""]
        for n in m.get("notes", []):
            md.append(f"- `{n}`")
        md.append("")

        tr = OUT / m["model"] / "transcripts.md"
        if tr.exists():
            body = tr.read_text()
            parts.append("<div class=transcript>" + _mini_md(body) + "</div>")
            md += [body, ""]
        for art in m["artifacts"]:
            parts.append(media_tag(m["model"], art))
            md.append(f"![{art['caption']}]({m['model']}/{art['file']}) — {art['caption']}")
        md.append("")

    if missing:
        parts.append("<h2>Not run</h2><p class=notes>" +
                     ", ".join(html.escape(x) for x in missing) + "</p>")

    (OUT / "index.html").write_text("\n".join(parts))
    (OUT / "index.md").write_text("\n".join(md))
    print(f"{(OUT / 'index.html').relative_to(ROOT)}  ({len(mans)} models)")
    if missing:
        print(f"  missing: {', '.join(missing)}")


def _mini_md(text):
    """Just enough markdown for the transcript fragments: ### and > lines."""
    out = []
    for line in text.splitlines():
        if line.startswith("### "):
            out.append(f"<b>{html.escape(line[4:])}</b>")
        elif line.startswith("> "):
            out.append(f"<blockquote>{html.escape(line[2:])}</blockquote>")
        elif line.startswith("_") and line.endswith("_"):
            out.append(f"<div class=notes>{html.escape(line.strip('_'))}</div>")
    return "\n".join(out)


if __name__ == "__main__":
    main()
