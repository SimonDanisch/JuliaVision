"""Read bounded native safetensors headers, never full checkpoint shards.

This is CPU/network preparation only. No torch/model imports, device access,
remote code execution, or activation of an alternate Julia environment.
"""
import hashlib
import json
import pathlib
import struct
import tomllib
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
LIMIT = 8_000_000


def fetch_prefix(url, count):
    if not 0 < count <= LIMIT:
        raise ValueError("bounded prefix required")
    request = urllib.request.Request(url, headers={"Range": f"bytes=0-{count-1}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        # Starting at zero remains correct if the server ignores Range. Never
        # read beyond the bounded prefix or stream a whole multi-GB shard.
        content_range = response.headers.get("Content-Range", "")
        if content_range and not content_range.startswith("bytes 0-"):
            raise ValueError("unexpected range offset")
        data = response.read(count)
    if len(data) != count:
        raise ValueError("incomplete bounded prefix")
    return data


def metadata(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        raw = response.read(LIMIT + 1)
    if len(raw) > LIMIT:
        raise ValueError("metadata exceeds bound")
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def audit():
    models = tomllib.loads((HERE / "models.toml").read_text())["models"]
    entry = next(row for row in models if row["id"] == "Qwen/Qwen3.6-35B-A3B")
    base = f'https://huggingface.co/{entry["id"]}/resolve/{entry["revision"]}'
    config, config_hash = metadata(base + "/config.json")
    index, index_hash = metadata(base + "/model.safetensors.index.json")
    c = config["text_config"]
    keys = ["lm_head.weight", "model.language_model.embed_tokens.weight"]
    prefix = "model.language_model.layers.0."
    keys += [prefix + name for name in (
        "mlp.experts.gate_up_proj", "mlp.experts.down_proj", "mlp.gate.weight",
        "mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight",
        "mlp.shared_expert.down_proj.weight", "mlp.shared_expert_gate.weight")]
    width, f, ne = c["hidden_size"], c["moe_intermediate_size"], c["num_experts"]
    expected = {
        keys[0]: [c["vocab_size"], width], keys[1]: [c["vocab_size"], width],
        prefix + "mlp.experts.gate_up_proj": [ne, 2 * f, width],
        prefix + "mlp.experts.down_proj": [ne, width, f],
        prefix + "mlp.gate.weight": [ne, width],
        prefix + "mlp.shared_expert.gate_proj.weight": [c["shared_expert_intermediate_size"], width],
        prefix + "mlp.shared_expert.up_proj.weight": [c["shared_expert_intermediate_size"], width],
        prefix + "mlp.shared_expert.down_proj.weight": [width, c["shared_expert_intermediate_size"]],
        prefix + "mlp.shared_expert_gate.weight": [1, width],
    }
    headers, rows = {}, []
    for key in keys:
        shard = index["weight_map"][key]
        if shard not in headers:
            size = struct.unpack("<Q", fetch_prefix(base + "/" + shard, 8))[0]
            if not 0 < size <= LIMIT - 8:
                raise ValueError("invalid safetensors header size")
            raw = fetch_prefix(base + "/" + shard, size + 8)[8:]
            headers[shard] = json.loads(raw)
        tensor = headers[shard][key]
        if tensor["shape"] != expected[key] or tensor["dtype"] != "BF16":
            raise ValueError(f"unexpected native tensor: {key}: {tensor}")
        rows.append(dict(key=key, shard=shard, shape=tensor["shape"], dtype=tensor["dtype"]))
        print(key, tensor["shape"], tensor["dtype"], flush=True)
    report = dict(scope="bounded native header/layout audit only, not full-model correctness or throughput",
                  model=entry, config_sha256=config_hash, index_sha256=index_hash,
                  shards_inspected=len(headers), rows=rows)
    output = HERE / "moe_native_headers.json"
    if output.exists():
        raise FileExistsError("audit already saved; do not silently replace it")
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    audit()
