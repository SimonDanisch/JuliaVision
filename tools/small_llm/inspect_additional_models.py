"""Pin alternate architectures from official public metadata, without weights."""
import json
from audit_moe_headers import HERE, metadata


def inspect():
    rows = []
    for name in ("LiquidAI/LFM2.5-24B-A2B", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"):
        info, _ = metadata("https://huggingface.co/api/models/" + name)
        revision = info["sha"]
        config, config_hash = metadata(f"https://huggingface.co/{name}/resolve/{revision}/config.json")
        rows.append(dict(id=name, revision=revision, license=info.get("cardData", {}).get("license"),
                         config_sha256=config_hash, config=config,
                         scope="configuration only; not integrated, locally benchmarked, or license-reviewed"))
        print(name, revision, config.get("model_type"), flush=True)
    output = HERE / "additional_model_configs.json"
    if output.exists():
        raise FileExistsError("metadata snapshot already saved")
    output.write_text(json.dumps(dict(scope="official pinned metadata, no remote code or weights", rows=rows), indent=2) + "\n")


if __name__ == "__main__":
    inspect()
