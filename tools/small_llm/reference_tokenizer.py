"""CPU-only token-ID fixtures from the checkpoint's native Rust tokenizer."""
import argparse
import json
from pathlib import Path
from tokenizers import Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
tokenizer = Tokenizer.from_file(str(args.checkpoint / "tokenizer.json"))
texts = [
    "The capital of France is Paris. The capital of Germany is",
    "Café e\u0301 😀\n日本語", "a 12.3\n\n\t b", "I'm WE'RE she'd they'll I've",
    "Deutsch: Straße, Größe, weiß. العربية Ελληνικά 中文 한국어",
    "  leading and trailing  ", "\r\n\r\n", "a\u0308 o\u0308 u\u0308", "\u0000\u0001\t",
    "<|im_start|>user\nHi<|im_end|>\n", "<tool_call>{\"name\":\"test\"}</tool_call>", "",
]
rows = [{"text": text, "ids": tokenizer.encode(text, add_special_tokens=False).ids}
        for text in texts]
args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
print(f"Wrote {len(rows)} tokenizer references to {args.output}")
