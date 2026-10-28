"""Export Qwen3-Embedding-0.6B for Qwen3EmbeddingRunner.

    uv run tools/models.py fetch qwen3embedding
    uv run tools/export_qwen3embedding.py

The model is Qwen3's base decoder read at its final norm. What this writes to
`gen/graphs/qwen3embedding`:

* `qwen3embedding.json` and `weights.safetensors`: the decoder over token
  embeddings of any length 2..MAX_TOKENS (the graph's `t` symbol), fp16.
* `embed_tokens.safetensors`: the 151669x1024 input table. A text reads a few
  hundred rows of it, so the runner gathers them on the host and the table never
  reaches the device, as in Qwen-Image's text encoder.
* `vocab.json`, `merges.txt`, `added_tokens.json`: the tokenizer, in the layout
  `QwenTokenizer` reads. `added_tokens.json` does not ship with the checkpoint;
  it is written from `tokenizer_config.json`'s `added_tokens_decoder`.
* `reference.safetensors` + `reference.json`: token ids and embeddings from the
  model card's own recipe in fp32, for the runner's parity test.
"""
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.export import Dim

sys.path.insert(0, "tools")

import export_graphs as EG
from common import find_root
from export_horizon32b import prune

ROOT = find_root()
SRC = ROOT / "gen" / "qwen3embedding"
OUT = ROOT / "gen" / "graphs" / "qwen3embedding"
# Package summaries and docstrings, the texts this embeds, fit in far less; the
# model itself takes 32k. A longer text is truncated by the runner.
MAX_TOKENS = 512
DT = torch.float16

# The model card's retrieval instruction format. Queries carry it, documents do not.
TASK = "Given a description of a programming need, retrieve Julia packages and functions that address it"
QUERIES = ["macro for data wrangling", "ocean sim", "contact between deformable bodies"]
DOCUMENTS = [
    "A collection of macro functions which simplify the syntax required to operate on DataFrames.jl objects.",
    "Fast and friendly ocean-flavored Julia software for simulating incompressible fluid dynamics in "
    "Cartesian and spherical shell domains on CPUs and GPUs.",
    "Plots",
    # Long enough to land in a larger bucket than the others, so the runner's
    # bucketing and its choice of the kept row are under test too.
    " ".join(["Agents.jl is a pure Julia general-purpose framework for agent-based modeling (ABM): "
              "a computational simulation methodology where autonomous agents react to their "
              "environment (including other agents) given a predefined set of rules."] * 3),
]


def instructed(query):
    return f"Instruct: {TASK}\nQuery:{query}"


class StaticRotary(torch.nn.Module):
    """Qwen3's rotary tables for every position up to a bound, as constants,
    sliced to the input's length: positions are always `0..t-1`, so the table
    for a shorter text is a prefix of the table for a longer one, and computing
    it once is what lets the length stay a symbol."""

    def __init__(self, rotary, positions: int, dtype):
        super().__init__()
        with torch.no_grad():
            cos, sin = rotary(torch.zeros(1, dtype=dtype), torch.arange(positions).view(1, positions))
        self.tables = (cos.contiguous(), sin.contiguous())

    def forward(self, x, position_ids):
        n = position_ids.shape[-1]
        return tuple(table[:, :n].to(x.device) for table in self.tables)


class StaticQwen3Encoder(torch.nn.Module):
    """The decoder over `inputs_embeds`, no cache, no mask: attention is causal
    by itself, which is also why right padding is safe.

    The causal mask is built here, without position ids, and handed in as the
    model's mask mapping. Left to the model, it is built WITH the position ids
    the model derives, and position ids with no attention mask and no cache are
    transformers' cue to look for packed sequences: a `cumsum`/`ne`/`eq`/`index`
    construction in the graph, for one sequence that is never packed. This way
    the mask is the plain `arange`/`le` one Qwen-Image's encoder carries."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs_embeds):
        from transformers.masking_utils import create_causal_mask
        mask = create_causal_mask(config=self.model.config, inputs_embeds=inputs_embeds,
                                  attention_mask=None, past_key_values=None, position_ids=None)
        return self.model(inputs_embeds=inputs_embeds, attention_mask={"full_attention": mask},
                          use_cache=False).last_hidden_state


def load(dtype):
    from transformers import AutoModel
    # `sdpa`: `eager` writes the causal mask as an additive tensor the graph would carry.
    return AutoModel.from_pretrained(SRC, dtype=dtype, attn_implementation="sdpa").eval()


def export():
    model = load(DT)
    # One row past the bound: a slice to a table's whole length is a case
    # `torch.export` guards against (see export_qwenimage21.py).
    model.rotary_emb = StaticRotary(model.rotary_emb, MAX_TOKENS + 1, DT)
    wrapper = StaticQwen3Encoder(model).eval()
    example = torch.zeros(1, 16, model.config.hidden_size, dtype=DT)
    # `Dim.DYNAMIC`, as for Qwen-Image's encoder: a named `Dim` must PROVE the
    # attention's contiguity guards over its range, which the solver cannot do.
    # The floor is 2 because `torch.export` specialises a dimension it can prove is 1.
    t = Dim.DYNAMIC(min=2, max=MAX_TOKENS)
    with torch.no_grad():
        program = torch.export.export(wrapper, (example,), dynamic_shapes=({1: t},),
                                      strict=False).run_decompositions()
    graph = EG.convert(program, ({1: "t"},), "qwen3embedding")
    available = {**dict(wrapper.state_dict()), **EG.save_constants(program)}
    tensors = {}
    prune(graph, available, tensors)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "qwen3embedding.json").write_text(json.dumps(graph, indent=1))
    save_file({k: v.detach().contiguous() for k, v in tensors.items()}, str(OUT / "weights.safetensors"))
    save_file({"embed_tokens": model.embed_tokens.weight.detach().contiguous()},
              str(OUT / "embed_tokens.safetensors"))
    histogram = {}
    for op in graph["ops"]:
        histogram[op["aten"]] = histogram.get(op["aten"], 0) + 1
    (OUT / "op_histogram.json").write_text(json.dumps(histogram, indent=1, sort_keys=True))
    cfg = model.config
    (OUT / "export.json").write_text(json.dumps({
        "model": "Qwen/Qwen3-Embedding-0.6B", "token_symbol": "t", "min_tokens": 2,
        "max_tokens": MAX_TOKENS, "hidden_size": cfg.hidden_size, "layers": cfg.num_hidden_layers,
        "endoftext": 151643, "task": TASK, "dtype": "float16"}, indent=1))
    print(f"qwen3embedding: {len(graph['ops'])} ops, {len(tensors)} weights, 2..{MAX_TOKENS} tokens")


def tokenizer_assets():
    for f in ("vocab.json", "merges.txt"):
        shutil.copy(SRC / f, OUT / f)
    config = json.loads((SRC / "tokenizer_config.json").read_text())
    added = {v["content"]: int(k) for k, v in config["added_tokens_decoder"].items()}
    (OUT / "added_tokens.json").write_text(json.dumps(added, indent=1))


def reference():
    """The model card's recipe in fp32: the tokenizer's post-processor appends
    `<|endoftext|>`, the hidden state at that last token is the embedding,
    L2-normalized."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(SRC)
    model = load(torch.float32)
    texts = [instructed(q) for q in QUERIES] + DOCUMENTS
    ids, rows = [], []
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt")
            assert int(enc.input_ids[0, -1]) == 151643, "the post-processor no longer appends <|endoftext|>"
            hidden = model(**enc).last_hidden_state[0, -1]
            rows.append(torch.nn.functional.normalize(hidden, dim=0))
            ids.append(enc.input_ids[0].to(torch.int32))
    lengths = torch.tensor([len(i) for i in ids], dtype=torch.int32)
    save_file({"ids": torch.cat(ids), "lengths": lengths, "embeddings": torch.stack(rows)},
              str(OUT / "reference.safetensors"))
    (OUT / "reference.json").write_text(json.dumps({"texts": texts, "queries": len(QUERIES)}, indent=1))
    sims = torch.stack(rows[:len(QUERIES)]) @ torch.stack(rows[len(QUERIES):]).T
    print("query x document cosine (fp32 reference):")
    for q, row in zip(QUERIES, sims):
        print(f"  {q!r:40} " + "  ".join(f"{s:.3f}" for s in row.tolist()))


if __name__ == "__main__":
    export()
    tokenizer_assets()
    reference()
