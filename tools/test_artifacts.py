"""Regression test for the developer-side artifacts in `tools/Artifacts.toml`.

    python3 tools/test_artifacts.py

Standard library only, and nothing is downloaded.

The bug this pins: the exporters, reference dumps and PyTorch benchmark found
their upstream repositories as `find_root() / "dev" / <name>`, and a `gen`
symlink inside the repository made `find_root` return the repository itself. So
five clones and a gigabyte of checkpoints were cloned into JuliaVision, unpinned,
and no other machine had them. They are artifacts now; these checks keep it so.
"""

import re
import sys
import tomllib
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parent
sys.path.insert(0, str(TOOLS))

from artifacts import TOML, depots, installed, treehash  # noqa: E402

RELEASE = "https://github.com/SimonDanisch/JuliaVision/releases/download/"


def bindings():
    with open(TOML, "rb") as fh:
        return tomllib.load(fh)


def test_every_binding_is_published():
    # Bound but only local is the failure `make_artifacts.jl` warns about: it
    # resolves here, because `create_artifact` put the tree in this store, and
    # fails on every other machine.
    for name, entry in bindings().items():
        assert entry.get("lazy") is True, f"{name} is not lazy"
        (download,) = entry["download"]
        assert download["url"].startswith(RELEASE), f"{name} downloads from {download['url']}"
        assert download["url"].endswith(f"/{name}.tar.gz"), f"{name} downloads {download['url']}"
        assert re.fullmatch(r"[0-9a-f]{64}", download["sha256"]), f"{name} sha256"
        assert re.fullmatch(r"[0-9a-f]{40}", entry["git-tree-sha1"]), f"{name} tree hash"


def test_everything_packed_for_tools_is_bound():
    # `make_artifacts.jl` binds into `tools/Artifacts.toml` every source in
    # `SOURCES` and every checkpoint in `CHECKPOINTS` whose owner is "tools".
    # A name in those tables that is missing here was packed and never bound,
    # or its binding was lost.
    src = (TOOLS / "make_artifacts.jl").read_text()
    sources = re.search(r"const SOURCES = Dict\((.*?)\n\)", src, re.S).group(1)
    checkpoints = re.search(r"const CHECKPOINTS = Dict\((.*?)\n\)", src, re.S).group(1)
    want = set(re.findall(r'"([\w.-]+)"\s*=>\s*\(', sources))
    want |= set(re.findall(r'"([\w.-]+)"\s*=>\s*\("tools"', checkpoints))
    assert {"sam2-src", "matanyone-src", "sam2-large-ckpt", "rife-ckpt"} <= want, want
    # `basicvsrpp-ckpt` is in the table for when its checkpoint is packed, and
    # has not been yet.
    missing = want - set(bindings()) - {"basicvsrpp-ckpt"}
    assert not missing, f"packed for tools/ but not bound: {sorted(missing)}"


def test_nothing_resolves_inside_the_repository():
    for depot in depots():
        assert not depot.resolve().is_relative_to(REPO), f"depot {depot} is inside {REPO}"
    for name in bindings():
        treehash(name)   # every binding is readable
        path = installed(name)
        if path is not None:
            assert not path.resolve().is_relative_to(REPO), f"{name} resolved to {path}"


# Each upstream that has a `-src` artifact, by the directory name its clone had.
CLONES = ["sam2", "MatAnyone2", "Depth-Anything-V2", "Practical-RIFE", "diffusers"]


def test_no_script_builds_a_clone_path():
    # How the bug came in: `ROOT / "dev" / "sam2"`, `Path("dev/diffusers/src")`.
    names = "|".join(re.escape(c) for c in CLONES)
    pattern = re.compile(rf"""["']?dev["']?\s*/\s*["']?({names})\b""")
    hits = []
    for path in sorted([*TOOLS.glob("*.py"), *(REPO / "benchmark").glob("*.py")]):
        if path.name == Path(__file__).name:
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if pattern.search(line):
                hits.append(f"{path.relative_to(REPO)}:{i}: {line.strip()}")
    assert not hits, "upstream source must come from its -src artifact:\n" + "\n".join(hits)


if __name__ == "__main__":
    tests = [f for n, f in sorted(globals().items()) if n.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} passed")
