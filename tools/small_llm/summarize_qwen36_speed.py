"""Summarize all retained samples; never drop slower or contended samples."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def summarize(path, feature="down_rows2"):
    data = json.loads(path.read_text())
    if not data["complete"] or not data["normal_final_waitidle"]:
        raise ValueError("Incomplete run or abnormal cleanup")
    groups = {"original": [], "final": [], "llama_cpp": []}
    for case in data["cases"]:
        group = "llama_cpp" if case["flags"] is None else "final" if feature in case["flags"] else "original"
        for sample in case["samples"]:
            if not sample["same_generated_sequence"]:
                raise ValueError("Generated sequence changed")
            if group != "llama_cpp":
                if not (sample["prompt_logits_bitidentical"] and sample["final_logits_bitidentical"]
                        and sample["submissions"] == data["tokens"]):
                    raise ValueError("Numerical/submission regression")
            groups[group].append(sample["tokens_per_second"])
    stats = {}
    for name, values in groups.items():
        quartiles = statistics.quantiles(values, n=4)
        stats[name] = dict(n=len(values), median=statistics.median(values),
                           minimum=min(values), maximum=max(values), q1=quartiles[0], q3=quartiles[2],
                           samples=values)
    return dict(source=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), feature=feature,
                checkpoint_sha256=data["checkpoint_sha256"], statistics=stats,
                final_vs_native_percent=100 * (stats["final"]["median"] / stats["llama_cpp"]["median"] - 1),
                observed_worst_final_vs_best_native_percent=100 * (stats["final"]["minimum"] / stats["llama_cpp"]["maximum"] - 1),
                final_vs_original_percent=100 * (stats["final"]["median"] / stats["original"]["median"] - 1),
                scope="One fixed 8-token prompt/64 greedy decode tokens, F32 KV, capacity256, same Radeon and GGUF; not broad model/context superiority",
                feedback_modes="Mantle device-resident token feedback; llama.cpp synchronized host logits/greedy feedback",
                contention_possible=data["potential_contention"], all_samples_retained=True,
                isolated_model_residency=data.get("isolate_external_model_residency", False),
                checked_logits_bitidentical_to_original_mantle=True,
                numerical_equivalence_to_llama_cpp_proven=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--feature", default="down_rows2")
    args = parser.parse_args()
    result = summarize(args.input, args.feature)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))
