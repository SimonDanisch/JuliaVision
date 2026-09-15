"""Tokenise / detokenise for the Julia runner, over JSON on stdin/stdout.

The generation happens in Julia; only the tokenizer lives here. `tokenizers` has
no Julia port that reads this checkpoint's `tokenizer.json`, and writing one to
print a sentence is not the point of the port.

    echo '{"encode": "hello"}'      -> {"ids": [...]}
    echo '{"decode": [1,2,3]}'      -> {"text": "..."}
    echo '{"chat": "hi"}'           -> {"ids": [...]}   # through the chat template
"""
import json
import sys

from transformers import AutoTokenizer

from common import find_root

tok = AutoTokenizer.from_pretrained(find_root() / "gen" / "horizon32b",
                                    trust_remote_code=True)
req = json.load(sys.stdin)
if "encode" in req:
    out = {"ids": tok(req["encode"], add_special_tokens=True)["input_ids"]}
elif "chat" in req:
    enc = tok.apply_chat_template(
        [{"role": "user", "content": req["chat"]}],
        add_generation_prompt=True, tokenize=True)
    # Newer transformers hands back a `BatchEncoding` here, older a plain list.
    out = {"ids": list(enc["input_ids"]) if hasattr(enc, "keys") else list(enc)}
else:
    out = {"text": tok.decode(req["decode"], skip_special_tokens=False)}
out["eos"] = tok.eos_token_id
json.dump(out, sys.stdout)
