"""misaki's phonemisation of a sentence corpus, as the parity reference.

    uv run tools/g2p_reference.py

`KokoroRunner`'s G2P is a Julia port of `misaki/en.py`. A port is worth exactly
what it agrees with, so this writes what misaki produces for a fixed corpus and
the Julia test suite diffs against it token by token.

**The fallback is deliberately off.** misaki hands out-of-vocabulary words to
espeak, a whole second G2P with its own alphabet; the Julia port has no espeak
and drops those words with a warning instead. Including espeak's output here
would score the port against a component it does not have and cannot get, so
`fallback=None` and the unknowns are marked `❓` in the reference — the Julia
side is expected to skip exactly those and nothing else.

The corpus mixes the cases the port has to get right beyond plain lookup:
context-sensitive articles (`the apple` vs `the pear`), inflections that only the
stemmers reach, numbers in several readings, acronyms, and possessives.
"""

import json
import re
from pathlib import Path

from common import find_root

ROOT = find_root()
OUT = ROOT / "gen" / "graphs" / "kokoro-dyn" / "g2p_reference.json"

CORPUS = [
    # Plain lookup, and the ordinary shape of a sentence.
    "Hello world, this is a test.",
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
    "I have no idea what you are talking about.",
    "Please remain seated until the aircraft comes to a complete stop.",
    "It was the best of times, it was the worst of times.",
    # `the` and `to` before a vowel versus a consonant — the context rule, which
    # is the one part of the port that has to look at the NEXT token.
    "The apple and the pear are on the table.",
    "The engine, the answer, the object, the union.",
    "I want to eat and to sleep and to open the door.",
    "Go to school, to a party, to England, to the office.",
    "An hour, an apple, a house, a union, an honest mistake.",
    # The stemmers: plural, past, progressive, each with its voicing rule.
    "The cats and the dogs and the buses and the churches.",
    "He walked and talked and needed and wanted and painted.",
    "She is running, writing, sitting, hoping, and swimming.",
    "The children's toys and the students' books and James's car.",
    "They realized the batteries were completely discharged.",
    "Photographers photographed the photographs photographically.",
    # Numbers, in every reading the code distinguishes.
    "There are 42 apples and 7 oranges.",
    "In 1984 the population reached 1,250,000 people.",
    "The temperature is 21.5 degrees at 3 o'clock.",
    "He came 1st, she came 2nd, and they came 23rd.",
    "Chapter 100, page 365, room 12.",
    "The year 2000 and the year 2007 and the year 1900.",
    # Acronyms and capitalisation.
    "NASA and the FBI and the CIA held a meeting.",
    "The USA, the UK, and the EU signed the agreement.",
    "Dr. Smith met Mr. Jones on Monday.",
    # Punctuation, which Kokoro renders as pauses.
    "Wait — what? No! Really; yes, of course...",
    "First, second, and third: that is the order.",
    # Longer, ordinary prose.
    "The neural network processes audio at twenty four kilohertz.",
    "Julia compiles to native code through a just in time compiler.",
    "Speech synthesis converts written language into spoken words.",
    "This sentence was generated entirely on the graphics processor.",
    "Machine learning models require substantial computational resources.",
]


def numbers():
    """`num2words` for every reading the G2P produces, as word lists.

    Split the way misaki splits it — `re.split(r'[^a-z]+')`, then drop `and` —
    because that is what reaches the lexicon. Comparing the raw strings would
    fail on hyphens that never get looked up.

    **Every four-digit year, not a sample.** The year reading has a rule that is
    easy to get subtly wrong (2007 is "two thousand seven" but 2024 is "twenty
    twenty four" and 1101 is "eleven oh one"), and a hand-picked set of examples
    is exactly how a wrong rule survives.
    """
    from num2words import num2words

    def words(s):
        return [w for w in re.split(r"[^a-z]+", s) if w and w != "and"]

    return {
        "cardinal": {str(n): words(num2words(n))
                     for n in [*range(0, 130), *range(130, 1000, 7),
                               *range(1000, 100000, 331), 1000000, 1234567]},
        "ordinal": {str(n): words(num2words(n, to="ordinal"))
                    for n in [*range(1, 130), *range(130, 1000, 11), 1000, 1000000]},
        "year": {str(n): words(num2words(n, to="year")) for n in range(1000, 10000)},
    }


def main(out: Path):
    from misaki import en
    g = en.G2P(trf=False, british=False, fallback=None, unk="❓")
    rows = []
    nunk = 0
    for text in CORPUS:
        ps, _ = g(text)
        nunk += ps.count("❓")
        rows.append({"text": text, "phonemes": ps})
    nums = numbers()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"sentences": rows, "numbers": nums},
                              indent=1, ensure_ascii=False))
    print(f"  {len(rows)} sentences, {nunk} out-of-vocabulary tokens")
    print(f"  numbers: {sum(len(v) for v in nums.values())} readings "
          f"({len(nums['year'])} years) -> {out}")


if __name__ == "__main__":
    main(OUT)
