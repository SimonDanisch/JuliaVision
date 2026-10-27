"""Necessary traffic bounds, NOT measured throughput or full roofline predictions."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def dense_weight_floor(parameters, bits, tokens_per_second):
    # Count weights actually scanned, not sparse embedding lookups. Optimistic:
    # no scales/zeros, KV, scratch or rereads.
    return parameters * bits / 8 * tokens_per_second / 1e9


def kv_bytes(config, context, bytes_per_element):
    layers = config["layer_types"].count("full_attention")
    return 2 * layers * config["num_key_value_heads"] * config["head_dim"] * context * bytes_per_element


def report():
    config = json.loads((ROOT / "gen/qwen35-4b/config.json").read_text())["text_config"]
    rows = [dict(scanned_weight_parameters_billions=size, bits=bits, tokens_per_second=rate,
                 minimum_weight_GB_per_second=dense_weight_floor(size * 1e9, bits, rate))
            for size in (4, 12, 20) for bits in (4, 16) for rate in (50, 100)]
    caches = [dict(context=context, bytes_per_element=element,
                   KV_GiB=kv_bytes(config, context, element) / 2**30,
                   minimum_KV_GB_per_second_at_50=kv_bytes(config, context, element) * 50 / 1e9)
              for context in (4096, 262144, 1000000) for element in (2, 4)]
    assert dense_weight_floor(12e9, 4, 50) == 300
    assert dense_weight_floor(20e9, 4, 50) == 500
    assert kv_bytes(config, 1000000, 2) == 32768000000
    output = dict(scope="analytic necessary traffic bounds, NOT measured performance",
                  assumptions=["Batch-one decode scans the stated number of weights per token; optimistic ideal bit rate",
                               "Scanned weights are NOT automatically total model parameters: sparse input-embedding lookups must be excluded",
                               "KV traffic assumes exact full-context attention, one read of K and V per step",
                               "No hardware bandwidth claim or throughput prediction is made",
                               "MoE active-weight traffic differs; total parameter count is not active count",
                               "Sparse attention, cache compression, batching and alternative memory architectures can change these assumptions"],
                  dense_weights=rows, qwen35_4b_attention_KV=caches)
    (HERE / "bandwidth_floor.json").write_text(json.dumps(output, indent=2) + "\n")
    return output


if __name__ == "__main__":
    report()
