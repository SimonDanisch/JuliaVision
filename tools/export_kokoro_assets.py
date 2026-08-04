"""Everything Kokoro needs on the HOST, as data the Julia package can read.

    uv run tools/export_kokoro_assets.py

The graph exporter writes weights and ops. This writes the three tables that sit
*outside* the graph and would otherwise pin the runner to a Python install:

  * **`vocab.json`** — `KModel.vocab`, phoneme character -> token id. 178 entries.
    The graph's first op is an embedding lookup; without this there is nothing to
    look up.
  * **`voices.safetensors`** — all 54 voice packs, each `(511, 1, 256)`.
    **Indexed by token count**, not a single vector per voice: Kokoro conditions
    prosody on how long the utterance is, and `KPipeline` takes `pack[len(ps)-1]`.
    Getting that wrong is silent — it produces fluent speech in a subtly wrong
    cadence rather than an error.
  * **`lexicon.json`** — misaki's US gold and silver lexicons, so the Julia G2P
    needs no Python at run time. Kept as two tables, not one: the lookup
    consults them differently (see [`lexicon`](#lexicon)).

## The POS variants are kept, and the heteronyms are what they cost

790 of the gold entries are POS-conditioned: `{'DEFAULT': ..., 'NOUN': ...}` —
`read`, `lead`, `live`, `wind`. misaki disambiguates them with a spaCy tagger,
which is a second model and a second dependency. The variants are written out
under `word|POS` keys alongside the default, so the Julia side *can* use them if
it ever learns the tag, and takes `DEFAULT` today. Both sides choose from the
same table either way, so a disagreement is a tagging disagreement rather than a
lexicon one.
"""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from common import find_root

ROOT = find_root()
OUT = ROOT / "gen" / "graphs" / "kokoro-dyn"


def grow(d):
    """misaki's `Lexicon.grow_dictionary`, verbatim.

    Adds the capitalised form of every lowercase entry and the lowercase form of
    every capitalised one. Without it `The` misses while `the` hits, and
    sentence-initial words fall through to the fallback — which is a different
    G2P and a visibly different pronunciation.
    """
    e = {}
    for k, v in d.items():
        if len(k) < 2:
            continue
        if k == k.lower():
            if k != k.capitalize():
                e[k.capitalize()] = v
        elif k == k.lower().capitalize():
            e[k.lower()] = v
    return {**e, **d}


def lexicon():
    """misaki's US lexicons, as `{"gold": ..., "silver": ...}`.

    **The two halves stay separate** because the lookup treats them differently:
    an all-caps word not found in gold is retried as a proper noun and *never*
    consults silver. Merging them changes which words get spelled out letter by
    letter.

    POS-conditioned entries (790 of them: `read`, `lead`, `live`, `wind`) are
    written out under `word|POS` keys alongside their `DEFAULT`. misaki picks
    between them with a spaCy tagger; the Julia side takes `DEFAULT` and can use
    the rest if it ever learns the tag. Either way both sides choose from the
    same table, so a disagreement is a tagging disagreement, not a lexicon one.
    """
    import misaki
    data = Path(misaki.__file__).parent / "data"
    out, npos = {}, 0
    for half, name in (("gold", "us_gold.json"), ("silver", "us_silver.json")):
        tbl = {}
        for word, val in grow(json.loads((data / name).read_text())).items():
            if isinstance(val, str):
                tbl[word] = val
            elif isinstance(val, dict):
                npos += 1
                for pos, ps in val.items():
                    if ps is None:
                        continue
                    tbl[word if pos == "DEFAULT" else f"{word}|{pos}"] = ps
        out[half] = tbl
    return out, npos


def voices(dev="cpu"):
    """All 54 packs, `(511, 1, 256)` each, keyed by voice name."""
    from huggingface_hub import hf_hub_download, list_repo_files
    names = [f for f in list_repo_files("hexgrad/Kokoro-82M") if f.startswith("voices/")]
    out = {}
    for f in sorted(names):
        p = hf_hub_download("hexgrad/Kokoro-82M", f)
        v = torch.load(p, map_location=dev, weights_only=True)
        assert v.shape[1:] == (1, 256), f"{f}: {tuple(v.shape)}"
        out[Path(f).stem] = v.squeeze(1).contiguous()      # (511, 256)
    return out


def main(out: Path):
    from kokoro import KModel
    gen = ROOT / "gen" / "kokoro"
    model = KModel(repo_id="hexgrad/Kokoro-82M",
                   config=str(gen / "config.json"),
                   model=str(gen / "kokoro-v1_0.pth")).eval()

    out.mkdir(parents=True, exist_ok=True)
    (out / "vocab.json").write_text(json.dumps(model.vocab, indent=1, sort_keys=True))
    print(f"  vocab.json: {len(model.vocab)} phoneme ids")

    packs = voices()
    save_file(packs, str(out / "voices.safetensors"))
    nb = sum(v.numel() * v.element_size() for v in packs.values())
    print(f"  voices.safetensors: {len(packs)} voices, {nb / 2**20:.1f} MiB")

    lex, npos = lexicon()
    (out / "lexicon.json").write_text(json.dumps(lex, sort_keys=True))
    print(f"  lexicon.json: {len(lex['gold'])} gold + {len(lex['silver'])} silver ({npos} POS-conditioned), "
          f"{(out / 'lexicon.json').stat().st_size / 2**20:.1f} MiB")


if __name__ == "__main__":
    main(OUT)
