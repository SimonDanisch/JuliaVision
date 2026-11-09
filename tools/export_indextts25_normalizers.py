"""Export upstream text-normalization transducers for the Julia interpreter.

Run in the IndexTTS Python environment. Python/Pynini are export dependencies;
the runner consumes only the resulting numeric tables.
"""

import json
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from common import find_root

REPO = Path.home() / "tools" / "index-tts"
sys.path.insert(0, str(REPO))


def export_fst(fst, path):
    offsets, finals, ilabels, olabels, weights, targets = [0], [], [], [], [], []
    for state in fst.states():
        finals.append(float(fst.final(state)))
        for arc in fst.arcs(state):
            ilabels.append(arc.ilabel)
            olabels.append(arc.olabel)
            weights.append(float(arc.weight))
            targets.append(arc.nextstate)
        offsets.append(len(ilabels))
    save_file(dict(start=np.array([fst.start()], dtype=np.int32),
                   offsets=np.array(offsets, dtype=np.int64),
                   finals=np.array(finals, dtype=np.float32),
                   ilabels=np.array(ilabels, dtype=np.int32),
                   olabels=np.array(olabels, dtype=np.int32),
                   weights=np.array(weights, dtype=np.float32),
                   targets=np.array(targets, dtype=np.int32)), str(path))
    print(f"{path.name}: {len(finals)} states, {len(ilabels)} arcs", flush=True)


def main():
    from indextts.utils.front import TextNormalizer
    frontend = TextNormalizer()
    frontend.load()
    out = find_root() / "gen" / "indextts25" / "graphs" / "normalizer"
    out.mkdir(parents=True, exist_ok=True)
    for lang, normalizer in [("en", frontend.en_normalizer), ("zh", frontend.zh_normalizer)]:
        export_fst(normalizer.tagger, out / f"{lang}_tagger.safetensors")
        export_fst(normalizer.verbalizer, out / f"{lang}_verbalizer.safetensors")
    sentences = [
        "The quick brown fox jumps over the lazy dog, and nobody seems to mind.",
        "I have 42 apples and 3.5 oranges.", "It costs $12.50, or 25% less.",
        "Dr. Smith arrived at 14:35 on 12/03/2024.", "What's GPT-5-nano?",
        "Read <lead|LEED> and <行|XING2>.", "Hello, world!", "one two three",
        "今天是2024年3月12日，价格是12.5元。", "欢迎大家来体验indextts2。",
        "百分之25，电话135-4567-8900。", "GPT-5-nano和F5-TTS很好用。",
        "克里斯托弗·诺兰说：你好！", "请读xuan4和ju3。", "12345",
    ]
    refs = []
    for text in sentences:
        clean = frontend.clean_pattern.sub(lambda m: frontend.char_rep_map[m.group()], text)
        refs.append(dict(text=text, clean=clean, normalized=frontend.normalize(clean)))
    for lang, normalizer in [("en", frontend.en_normalizer), ("zh", frontend.zh_normalizer)]:
        for text in sentences:
            tagged = normalizer.tag(text)
            refs.append(dict(lang=lang, text=text, tagged=tagged,
                             normalized=normalizer.verbalize(tagged)))
    (out / "reference.json").write_text(json.dumps(refs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
