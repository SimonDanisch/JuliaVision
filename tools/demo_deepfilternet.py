# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "deepfilternet==0.5.6",
#     "soundfile",
#     "torch",
#     "torchaudio",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# torchaudio = { index = "pytorch-cpu" }
# ///
"""The DeepFilterNet demo, in the one environment DeepFilterNet installs into.

    uv run tools/demo_deepfilternet.py      # or: uv run tools/demos.py deepfilternet

It cannot live in the project's: `deepfilternet` 0.5.6 (the last release, 2023)
requires `numpy<2` and `packaging<24`, and its `deepfilterlib` has wheels only up
to CPython 3.11, so on the project's 3.12 it is built from Rust source. As a
project dependency it made every `uv run` try that build and pinned the whole
lock to numpy 1.26.

torch and torchaudio come from the CPU index and nothing else does: that index
also carries a `packaging`, and taking it from there leaves no version 0.5.6 accepts.
"""

import demos

demos.deepfilternet()
