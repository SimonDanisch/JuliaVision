"""The developer-side inputs — upstream source and upstream checkpoints — as Julia
artifacts, resolved from Python.

    from artifacts import artifact
    sam2 = artifact("sam2-src")          # the upstream repository at its pinned commit
    ckpt = artifact("sam2-large-ckpt")   # the upstream checkpoint

They are bound in `tools/Artifacts.toml` and packed by `tools/make_artifacts.jl`
(`SOURCES` and `CHECKPOINTS`), the same way the runner packages bind what they
run. That makes every input of the exporters, the reference dumps and the PyTorch
benchmark content-addressed, pinned, and fetched on first use into the Julia
depot, so a fresh machine or a second workspace needs nothing cloned by hand.

These used to be `git clone`s under `<root>/dev/`, where `root` was whatever
`common.find_root` guessed. The `gen` symlink inside the repository made it guess
the repository itself, so five clones and a gigabyte of checkpoints ended up
inside JuliaVision, unpinned, and invisible to every other machine.

Resolution is split on purpose. The hit path is pure Python: an artifact lives
at `<depot>/artifacts/<git-tree-sha1>`, and the tree hash is in the TOML. A miss
asks `fetch_artifacts.jl` to install it, so the download is verified by `Pkg`
rather than by a second implementation of its checks written here.
"""

import os
import shutil
import subprocess
import tomllib
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
TOML = TOOLS / "Artifacts.toml"
FETCH = TOOLS / "fetch_artifacts.jl"


def depots():
    """The Julia depots to look in, in `JULIA_DEPOT_PATH` order.

    An empty entry stands for the default depot, as it does for Julia itself.
    """
    default = Path.home() / ".julia"
    env = os.environ.get("JULIA_DEPOT_PATH")
    if not env:
        return [default]
    return [Path(p) if p else default for p in env.split(os.pathsep)]


def treehash(name):
    with open(TOML, "rb") as fh:
        table = tomllib.load(fh)
    if name not in table:
        raise KeyError(f"no artifact {name!r} in {TOML}; known: {', '.join(sorted(table))}")
    return table[name]["git-tree-sha1"]


def installed(name):
    """The artifact's directory if some depot already has it, else `None`."""
    sha = treehash(name)
    for depot in depots():
        path = depot / "artifacts" / sha
        if path.is_dir():
            return path
    return None


def artifact(name):
    """Path to the artifact `name`, installing it on first use."""
    path = installed(name)
    if path is not None:
        return path
    julia = shutil.which("julia")
    if julia is None:
        raise SystemExit(f"artifact {name!r} is not installed and there is no `julia` on PATH "
                         f"to install it with; run `julia {FETCH} {name}`")
    out = subprocess.run([julia, "--startup-file=no", str(FETCH), name],
                         check=True, stdout=subprocess.PIPE, text=True).stdout
    path = Path(out.strip().splitlines()[-1])
    if not path.is_dir():
        raise RuntimeError(f"{FETCH} reported {path} for {name!r}, which is not a directory")
    return path
