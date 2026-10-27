"""Adapt original Python stdin/stdout tests to sandboxed Julia candidates.

Local Julia1.12/package contract, NOT the upstream Julia1.11 container score.
No reference solutions or tests are included in generation prompts.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time

from qualify_bigcodebench_x_reference import DRIVER, sandbox_command

ROOT = Path("/home/sim/Programmieren/VulkanDev")
JULIA_TREE = Path("/home/sim/.julia/juliaup/julia-1.12.7+0.x64.linux.gnu")
JULIA_COMMAND = [str(JULIA_TREE / "bin/julia"), "--startup-file=no", "--compiled-modules=existing",
                 "--project=" + str(ROOT), "--compile=min", "-O0", "/input/program.jl"]
OLD_COMMAND = '[sys.executable,"/input/program.py"]'
if DRIVER.count(OLD_COMMAND) != 1:
    raise ValueError("Reference driver changed; qualify adapter again")
JULIA_DRIVER = DRIVER.replace(OLD_COMMAND, repr(JULIA_COMMAND)).replace("timeout=20", "timeout=45")


def julia_sandbox_command(task, environment):
    command = sandbox_command(task, environment)
    options = []
    mounts = [JULIA_TREE, ROOT / "Project.toml", ROOT / "Manifest.toml"]
    if (ROOT / "LocalPreferences.toml").exists():
        mounts.append(ROOT / "LocalPreferences.toml")
    mounts += [Path("/home/sim/.julia") / name for name in ("packages", "artifacts", "compiled")]
    mounts += [ROOT / "dev" / name for name in ("Makie", "KernelAbstractions", "Adapt", "CompilerCaching")]
    for path in mounts:
        options += ["--ro-bind", str(path), str(path)]
    atlas = Path("/home/sim/.julia/scratchspaces/ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a/makie")
    if atlas.exists():
        options += ["--ro-bind", str(atlas), "/work/depot/scratchspaces/ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a/makie"]
    if Path("/etc/fonts").exists():
        options += ["--ro-bind", "/etc/fonts", "/etc/fonts"]
    options += ["--setenv", "JULIA_DEPOT_PATH", "/work/depot:/home/sim/.julia:",
                "--setenv", "JULIA_NUM_THREADS", "2", "--setenv", "JULIA_GLFW_PLATFORM", "null"]
    command[-2:-2] = options
    return command


def evaluate_julia(program, test_suite, environment):
    if not isinstance(program, str) or not isinstance(test_suite, str):
        raise TypeError("Code/test strings required")
    if len(program.encode()) > 262144 or len(test_suite.encode()) > 262144:
        raise ValueError("Bounded source required")
    with tempfile.TemporaryDirectory(prefix="bcbx-julia-") as root:
        task = Path(root)
        (task / "program.jl").write_text(program)
        (task / "tests.py").write_text(test_suite)
        (task / "driver.py").write_text(JULIA_DRIVER)
        started = time.monotonic()
        result = subprocess.run(julia_sandbox_command(task, environment), capture_output=True, text=True, timeout=190)
        return {"passed": result.returncode == 0 and result.stdout.strip() == "REFERENCE_TESTS_PASS",
                "returncode": result.returncode, "seconds": time.monotonic() - started,
                "stdout_tail": result.stdout[-4096:], "stderr_tail": result.stderr[-4096:],
                "program_sha256": hashlib.sha256(program.encode()).hexdigest(),
                "test_suite_sha256": hashlib.sha256(test_suite.encode()).hexdigest(),
                "scope": "Local Julia functional stdin/stdout test result; not plot aesthetics or GPU performance"}
