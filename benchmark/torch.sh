#!/bin/bash
# Run something under the PyTorch that can actually see this card.
#
#     benchmark/torch.sh benchmark/torch_models.py --out benchmark/results/torch.json
#
# The project's own `.venv` has a CUDA wheel (`+cu128`) and `torch.cuda.is_available()`
# is False on this AMD box, so it is fine for exporting graphs and cannot time
# anything. The ROCm wheel lives in a separate interpreter, and this script is
# the only place its location is written down.
#
#   JULIAVISION_TORCH_PYTHON      the interpreter to use
#                                 (default: ../tmp/baseline/.venv/bin/python)
#   JULIAVISION_TORCH_PRELOAD     LD_PRELOAD for it (default: the system ROCm pair)
#   JULIAVISION_TORCH_EXTRA_SITE  site-packages to APPEND to sys.path
#                                 (default: the project .venv's)
#
# The ROCm interpreter has torch and little else, while the model setups need
# safetensors, transformers, kokoro and hydra — which the project's own .venv
# has, along with a CUDA torch. So the extra path is passed as a variable and
# APPENDED by the script after `import torch`, rather than exported as
# PYTHONPATH: PYTHONPATH is searched before the interpreter's own
# site-packages and would import the CUDA build into the ROCm process.
#
# **Why the preload.** The ROCm wheel bundles its own runtime and that one
# segfaults at the first kernel launch on this gfx1151 box; the system ROCm 7.1
# launches the same kernels fine, verified with a plain hipcc program. Preloading
# the system HSA and HIP runtimes is what makes the wheel usable here.
#
# The default interpreter is under `tmp/`, which is scratch and is not backed up.
# If it is gone, recreate it — `benchmark/README.md` has the commands — or point
# `JULIAVISION_TORCH_PYTHON` at one you already have.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
py="${JULIAVISION_TORCH_PYTHON:-$here/../tmp/baseline/.venv/bin/python}"
preload="${JULIAVISION_TORCH_PRELOAD:-/usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1:/usr/lib/x86_64-linux-gnu/libamdhip64.so.7}"

if [[ ! -x "$py" ]]; then
    cat >&2 <<EOF
benchmark/torch.sh: no interpreter at
    $py

That is the ROCm PyTorch the timing side needs, and it is not the project's own
.venv (which is a CUDA build and sees no device here). Either recreate it — see
benchmark/README.md — or set JULIAVISION_TORCH_PYTHON to one that has a
GPU-visible torch.
EOF
    exit 1
fi

default_site="$($here/../.venv/bin/python -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)"
export JULIAVISION_TORCH_EXTRA_SITE="${JULIAVISION_TORCH_EXTRA_SITE-$default_site}"
export LD_PRELOAD="$preload"
exec "$py" "$@"
