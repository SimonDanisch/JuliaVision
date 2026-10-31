"""CPU-only independent Jinja parity for actual frozen coding prompts.

Loads just the unchanged Julia render_chat function, not GPU packages/devices.
The reference is the pinned checkpoint's own Jinja template, not a rewrite.
"""
import hashlib
import json
import subprocess
from pathlib import Path

import jinja2

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[3]
config = WORKSPACE / "gen/qwen35-4b/tokenizer_config.json"
template = jinja2.Environment().from_string(json.loads(config.read_text())["chat_template"])

messages = []
for path in sorted((ROOT / "kernel_transfer_results").glob("transfer_*.json")):
    row = json.loads(path.read_text())
    if "messages" in row:
        messages.append([{"role": role, "content": content}
                         for item in row["messages"] for role, content in item.items()])
assert len(messages) == 8, "the frozen eight-output study must be complete"
messages.extend([
    [{"role": "user", "content": "  café e\u0301 😀\n "}],
    [{"role": "system", "content": " concise "},
     {"role": "user", "content": "first"},
     {"role": "assistant", "content": "<think>old reasoning</think>\n\nold answer"},
     {"role": "user", "content": "next"}],
    [{"role": "user", "content": "first"},
     {"role": "assistant", "content": "answer", "reasoning_content": " reason "}],
])
fixtures = [{"messages": chat, "thinking": thinking}
            for chat in messages for thinking in (False, True)]
code = r'''
using JSON3
path=ARGS[1]
source=read(path,String)
start=findfirst("function render_chat(",source).start
stop=findfirst("\nchat_ids(",source).start-1
include_string(Main,source[start:stop],path)
fixtures=JSON3.read(read(stdin,String))
JSON3.write(stdout,[render_chat(f.messages;thinking=f.thinking) for f in fixtures])
'''
process = subprocess.run(
    ["julia", "--project=.", "--startup-file=no", "-e", code, str(ROOT / "qwen_text.jl")],
    cwd=WORKSPACE, input=json.dumps(fixtures), text=True, capture_output=True, check=True,
)
actual = json.loads(process.stdout)
for index, (fixture, rendered) in enumerate(zip(fixtures, actual, strict=True)):
    reference = template.render(messages=fixture["messages"], add_generation_prompt=True,
                                enable_thinking=fixture["thinking"])
    assert rendered == reference, f"chat template mismatch at fixture {index}"
print(json.dumps({"complete": True, "passed": len(fixtures),
                  "scope": "CPU plain-text chat rendering only; not token or quality evidence",
                  "checkpoint_template_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                  "renderer_sha256": hashlib.sha256((ROOT / "qwen_text.jl").read_bytes()).hexdigest()}))
