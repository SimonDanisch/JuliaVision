"""Reference checksums for `DNNKernels/test/test_flowmatch.jl`, from diffusers itself.

    uv run tools/flowmatch_checksums.py

Runs diffusers' `FlowMatchEulerDiscreteScheduler` (and PyTorch's guidance
arithmetic) from the pinned `diffusers-src` artifact over fp16 inputs built from
integers, so the Julia test can build the same inputs bit for bit, and prints a
position-weighted checksum of each result.
"""
import sys, numpy as np, torch
sys.path.insert(0, "tools")
from artifacts import artifact
sys.path.insert(0, str(artifact("diffusers-src") / "src"))
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift

N = 4096
def fp16bits(mul, e0):
    return np.array([((((k * 40503) >> 5) & 1) << 15) | ((e0 + (k * 7) % 8) << 10) | ((k * mul) % 1024)
                     for k in range(N)], dtype=np.uint16)
def t16(bits): return torch.from_numpy(bits.view(np.float16).copy())
def checksum(t):
    w = {torch.float16: (torch.int16, 0xffff), torch.float32: (torch.int32, 0xffffffff)}[t.dtype]
    return sum((k + 1) * (int(b) & w[1]) for k, b in enumerate(t.contiguous().view(w[0]).tolist())) % 2**64

x, v = t16(fp16bits(2654435761, 11)), t16(fp16bits(2246822519, 12))
c, u = t16(fp16bits(3266489917, 10)), t16(fp16bits(668265263, 10))
print("inputs", checksum(x), checksum(v), checksum(c), checksum(u))
for sig, dt in [((0.8, 0.5), torch.float16), ((0.0, 1 / 49), torch.float16), ((0.8, 0.5), torch.float32)]:
    s = FlowMatchEulerDiscreteScheduler(shift=1.0)
    s.set_timesteps(sigmas=np.array(sig))
    out = s.step(v.to(dt), s.timesteps[0], x.to(dt), return_dict=False)[0]
    print("step", sig, dt, checksum(out))
for scale in (5.0, 3.3, 7.5):
    print("cfg", scale, checksum(u + scale * (c - u)))

qwen = dict(base_image_seq_len=256, base_shift=0.5, max_image_seq_len=8192, max_shift=0.9,
            num_train_timesteps=1000, shift=1.0, shift_terminal=0.02,
            time_shift_type="exponential", use_dynamic_shifting=True)
sig_all, ts_all = [], []
for n in (256, 1024, 4096, 5808, 6032, 6889, 494, 1665):
    for steps in range(2, 61):
        s = FlowMatchEulerDiscreteScheduler(**qwen)
        s.set_timesteps(sigmas=np.linspace(1.0, 1 / steps, steps), mu=calculate_shift(n, 256, 8192, 0.5, 0.9))
        sig_all.append(s.sigmas); ts_all.append(s.timesteps)
print("qwen sweep", checksum(torch.cat(sig_all)), checksum(torch.cat(ts_all)))
st = []
for steps in range(2, 61):
    s = FlowMatchEulerDiscreteScheduler(shift=5.0)
    s.set_timesteps(sigmas=np.linspace(1, 0, steps + 1)[:steps])
    st.append(s.sigmas)
print("static sweep", checksum(torch.cat(st)))
