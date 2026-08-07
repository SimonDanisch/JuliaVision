#!/bin/bash
# `ops.jl` is shared by every model, so a change there gets every suite, not just
# the one that motivated it. `clamp.default` appears 32x in whisper, 34x in
# sam2-large, 90x in wanvae, 6x in kokoro — and Kokoro's is the awkward one: its
# iSTFT clamps an INDEX with a SYMBOLIC bound, so the input is integer-typed and
# the result now converts on store instead of going through `coerce`.
#
# ALWAYS `--project=.` FROM THE PROJECT ROOT. Not `Pkg.test`, which resolves a
# separate test environment, and never `--project` inside a package directory: a
# package's own env lacks the vendored dev/Vulkan, so the device build throws
# before reaching the code under test. That has already produced two confidently
# wrong results in this repo (see the gpu_av_shaders note).
cd /sim/Programmieren/VideoEdit
for t in DNNKernels WhisperRunner KokoroRunner SAM2Runner MatAnyoneRunner; do
  f="dev/JuliaVision/$t/test/runtests.jl"
  if [ ! -f "$f" ]; then echo "=================== $t: no runtests.jl ==="; continue; fi
  echo "=================== $t ==================="
  timeout 2700 julia --project=. "$f" 2>&1 \
    | grep -E "Test Summary|\| *[0-9]+ +[0-9]+|Fail|Error|ERROR|error during" \
    | tail -14
  echo "   -> done $t"
done
echo "SWEEP_COMPLETE"
