"""Batch-one decode traffic estimator; no weights downloaded or GPU initialized.

Scenario envelopes, NOT benchmarks/confidence intervals. Dense and MoE active
traffic are separate from resident model size. Unknown architectures fail closed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

HERE = Path(__file__).resolve().parent
Q4 = 144 / 256  # Q4_K includes scales/mins, not ideal four bits
Q6 = 210 / 256
MODELS = (
    "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B", "Qwen/Qwen3-14B",
    "Qwen/Qwen3.8-27B", "Qwen/Qwen3.6-35B-A3B",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct", "Qwen/Qwen3-Coder-Next",
    "openai/gpt-oss-20b", "openai/gpt-oss-120b",
    "google/gemma-4-26B-A4B-it", "LiquidAI/LFM2-24B-A2B",
    "Qwen/Qwen3.8-Flash-Next",
)


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "small-llm-traffic-audit/1"})
    with urllib.request.urlopen(req, timeout=30) as response:
        data = response.read(2_000_001)
    if len(data) > 2_000_000:
        raise ValueError("Metadata only; refuse large response")
    return data


def snapshot(repo):
    api = json.loads(fetch("https://huggingface.co/api/models/" + repo))
    revision = api["sha"]
    url = f"https://huggingface.co/{repo}/resolve/{revision}/config.json"
    data = fetch(url)
    return dict(repo=repo, revision=revision, config_url=url,
                config_sha256=hashlib.sha256(data).hexdigest(), config=json.loads(data))


def matbytes(m, k, head=False):
    if k % 256:
        raise ValueError("Explicit quantization layout required")
    return m * k * (Q6 if head else Q4 if m >= 128 else 4)


def qwen_traffic(config):
    c = config.get("text_config", config)
    family = c["model_type"]
    hybrid = family in ("qwen3_5_text", "qwen3_5_moe_text", "qwen3_next")
    if not hybrid and family not in ("qwen3_moe", "qwen3"):
        raise ValueError("Unsupported Qwen architecture: " + family)
    h, layers = c["hidden_size"], c["num_hidden_layers"]
    hd = c["head_dim"]
    kinds = c.get("layer_types")
    if kinds is None:
        if hybrid:
            interval = c["full_attention_interval"]
            if not isinstance(interval,int) or isinstance(interval,bool) or interval <= 0:
                raise ValueError("Hybrid attention interval")
            kinds = ["full_attention" if (i+1)%interval==0 else "linear_attention" for i in range(layers)]
        else:
            kinds = ["full_attention"] * layers
    if len(kinds) != layers:
        raise ValueError("Layer layout")
    moe = "num_experts" in c
    active = resident = 0.
    for kind in kinds:
        # Common projections, excluding input embedding lookup and LM head.
        if kind == "linear_attention":
            kd = c["linear_num_key_heads"] * c["linear_key_head_dim"]
            vd = c["linear_num_value_heads"] * c["linear_value_head_dim"]
            nh = c["linear_num_value_heads"]
            projections = (matbytes(2*kd+vd, h) + matbytes(vd, h) +
                           matbytes(h, vd) + 2*matbytes(nh, h))
            projections += (2*kd+vd)*c["linear_conv_kernel_dim"]*4 + 2*nh*4 + c["linear_value_head_dim"]*4
        elif kind == "full_attention":
            q, kv = c["num_attention_heads"] * hd, c["num_key_value_heads"] * hd
            gate = c.get("attn_output_gate", family == "qwen3_next")
            projections = matbytes(q*(2 if gate else 1), h) + 2*matbytes(kv, h) + matbytes(h, q) + 2*hd*4
        else:
            raise ValueError("Unsupported attention kind")
        common = projections + 2*h*4
        if moe:
            f, experts, topk = c["moe_intermediate_size"], c["num_experts"], c["num_experts_per_tok"]
            one = 2*matbytes(f,h) + matbytes(h,f)
            sf = c.get("shared_expert_intermediate_size", 0)
            sh = (2*matbytes(sf,h)+matbytes(h,sf)+h*4) if sf else 0
            router = matbytes(experts,h)
            active += common + sh + router + topk*one
            resident += common + sh + router + experts*one
        else:
            f = c["intermediate_size"]
            common += 2*matbytes(f,h) + matbytes(h,f)
            active += common; resident += common
    head = matbytes(c["vocab_size"],h,head=True)
    # Match our current runner: F32 input embedding, independent Q6 output head.
    active += head + 2*h*4
    resident += head + c["vocab_size"]*h*4 + h*4
    full = kinds.count("full_attention")
    return dict(active_weight_bytes=active, resident_text_weight_bytes=resident,
                kv_bytes_per_position_f16=2*full*c["num_key_value_heads"]*hd*2,
                local_kv_fixed_bytes_f16=0, moe=moe, attention="full/GQA",
                method="Config-derived matrix dimensions; Q4_K projections/Q6_K head/F32 input lookup storage. Text only, excludes vision/MTP, allocator/prefill scratch and recurrent-state storage.")


def gpt_oss_traffic(c, native=False):
    h, layers, experts, topk = c["hidden_size"], c["num_hidden_layers"], c["num_local_experts"], c["experts_per_token"]
    expert_params = 3*h*c["intermediate_size"]
    attn = h*c["head_dim"]*(2*c["num_attention_heads"]+2*c["num_key_value_heads"])
    head = h*c["vocab_size"]
    # Native MXFP4 applies to experts; shared attention/head stay BF16.
    eb, ab, hb = (4.25/8, 2., 2.) if native else (Q4,Q4,Q6)
    active = layers*(expert_params*topk*eb + attn*ab + h*experts*ab) + head*hb
    resident = layers*(expert_params*experts*eb + attn*ab + h*experts*ab) + head*hb + head*2
    full = c["layer_types"].count("full_attention")
    local = layers-full
    kv = 2*c["num_key_value_heads"]*c["head_dim"]*2
    return dict(active_weight_bytes=active,resident_text_weight_bytes=resident,
                kv_bytes_per_position_f16=full*kv,
                local_kv_fixed_bytes_f16=local*kv*c["sliding_window"],moe=True,attention="alternating full/window",
                method="Native MXFP4 expert + BF16 shared/head" if native else "Hypothetical Q4_K experts/shared + Q6_K output head; requires requantization and appropriate kernels")


def advertised_traffic(active_b, total_b, config, label):
    # Coarse estimates only where an architecture-specific counter is absent.
    c = config.get("text_config",config)
    head = c["hidden_size"]*c["vocab_size"]
    active = max(0,active_b*1e9-head)*Q4 + head*Q6
    resident = total_b*1e9*Q4 + head*(Q6-Q4)
    return dict(active_weight_bytes=active,resident_text_weight_bytes=resident,moe=True,
                kv_bytes_per_position_f16=None,local_kv_fixed_bytes_f16=None,attention="not modeled",
                method=label+"; approximate advertised counts with explicit Q6 head, NOT shape-derived; KV/novel operators not modeled")


def speed(weights, kv, low_bw=130e9, high_bw=180e9, extra_ms=(0,0)):
    if weights <= 0 or kv < 0 or not 0 < low_bw <= high_bw or not 0 <= extra_ms[0] <= extra_ms[1]:
        raise ValueError("Invalid traffic/bandwidth/latency")
    return [1/((weights+2.5*kv)/low_bw+extra_ms[1]/1000),
            1/((weights+kv)/high_bw+extra_ms[0]/1000)]


def scenarios(row):
    w = row["active_weight_bytes"]
    extra = (1,4) if row["moe"] else (0,0)
    out=[]
    for context in (4096,32768,131072,1000000):
        rate = row["kv_bytes_per_position_f16"]
        kv = None if rate is None else rate*context+row["local_kv_fixed_bytes_f16"]
        out.append(dict(context=context,fp16_cache_bytes=kv,
                        scenario_tps=None if kv is None else speed(w,kv,extra_ms=extra),
                        context_is_supported="Check model card/RoPE/engine; arithmetic size alone is NOT support"))
    return dict(**row,short_context_weight_only_tps=speed(w,0,extra_ms=extra),
                optimistic_256GBs_weight_only_ceiling=256e9/w,
                fits_32GiB_weights_only=row["resident_text_weight_bytes"]<32*2**30,
                fits_128GiB_weights_only=row["resident_text_weight_bytes"]<128*2**30,contexts=out)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--refresh",action="store_true");args=parser.parse_args()
    metadata=HERE/"model_speed_metadata.json"
    if args.refresh:
        with ThreadPoolExecutor(max_workers=4) as pool:
            snapshots=list(pool.map(snapshot,MODELS))
        metadata.write_text(json.dumps(snapshots,indent=2)+"\n")
    snapshots=json.loads(metadata.read_text());rows=[];unsupported=[]
    for s in snapshots:
        c=s["config"].get("text_config",s["config"]);repo=s["repo"]
        if repo.startswith("Qwen/"):
            try: row=qwen_traffic(s["config"])
            except ValueError as e:
                unsupported.append(dict(repo=repo,reason=str(e)));continue
        elif c["model_type"]=="gpt_oss":
            for native in (True,False):
                rows.append(dict(repo=repo,format="native" if native else "requantized",revision=s["revision"],**scenarios(gpt_oss_traffic(c,native))))
            continue
        elif repo.startswith("google/"):
            row=advertised_traffic(4,26,s["config"],"Gemma4 publisher rounded active/total counts")
        elif repo.startswith("LiquidAI/"):
            row=advertised_traffic(2.3,24,s["config"],"LFM2 publisher active/total counts")
        else:
            unsupported.append(dict(repo=repo,reason="No architecture model"));continue
        rows.append(dict(repo=repo,revision=s["revision"],format="Q4_K/Q6_head",**scenarios(row)))
    profile=json.loads((HERE/"qwen4b_decode_profile.json").read_text())
    benchmark=json.loads((HERE/"qwen4b_q4_resident_benchmark.json").read_text())
    result=dict(scope="CPU-only analytical scenarios, NOT measured larger-model throughput",hardware="Ryzen AI Max+395 / Radeon8060S / LPDDR5x8000 256-bit",
                peak_bandwidth_GBs=256,calibration=dict(nominal_weight_bytes=profile["inventory"]["nominal_weight_read_bytes"],
                measured_short_tps=benchmark["median_decode_tps"],effective_weight_GBs=profile["inventory"]["nominal_weight_read_bytes"]*benchmark["median_decode_tps"]/1e9),
                assumptions=["Batch-one nonspeculative decode, resident weights; no CPU/GPU offload over PCIe",
                    "Effective bandwidth scenario130..180GB/s calibrated loosely from historical4B weight traffic, not a separately measured memcpy bandwidth",
                    "KV read multiplier1..2.5; MoE additional1..4ms/token UNCALIBRATED architecture-transfer allowance, not a proven bound",
                    "Q4_K=4.5bits, Q6_K=6.5625bits. Actual quantization formats/quality/kernel efficiency change results",
                    "Weight-only memory-fit flags are NOT usable memory guarantees; add KV, recurrent state, scratch, OS and loader peaks",
                    "No cold-load/prefill/thinking-length/answer-quality predictions; benchmark successful tasks per second separately",
                    "Context rows do NOT imply native context or engine support. Gemma/LFM long-context estimates withheld rather than fabricating KV layout"],
                source_files_sha256={f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in
                    ("model_speed_metadata.json","model_speed_estimator.py","qwen4b_decode_profile.json",
                     "qwen4b_q4_resident_benchmark.json","qwen2b_q4_resident_benchmark.json")},
                dense_streamed_parameter_scenarios=[dict(scanned_parameters_billions=p,
                    stored_bits_per_parameter=4.5,active_weight_GB=p*Q4,
                    short_weight_only_tps=speed(p*1e9*Q4,0),
                    bandwidth_required_GBs_at_50=p*Q4*50,
                    optimistic_256GBs_ceiling=256/(p*Q4)) for p in (8,12,20,32,70)],
                rows=rows,unsupported=unsupported)
    output=HERE/"model_speed_estimates.json";output.write_text(json.dumps(result,indent=2)+"\n")
    for r in rows:
        print(r["repo"],r["format"],"activeGB",round(r["active_weight_bytes"]/1e9,2),"residentGiB",round(r["resident_text_weight_bytes"]/2**30,1),"short",[round(v,1) for v in r["short_context_weight_only_tps"]])
    print("WITHHELD",unsupported)


if __name__=="__main__":
    main()
