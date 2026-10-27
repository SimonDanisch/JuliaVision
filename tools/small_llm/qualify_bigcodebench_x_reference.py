"""Run pinned PUBLIC Python reference/tests in a CPU-only networkless jail.

This qualifies the external benchmark, NOT Julia/model answers. No generated
code or reference code executes in the orchestration process or host namespace.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

DRIVER = r'''
import importlib.util, pathlib, subprocess, sys, tempfile
spec=importlib.util.spec_from_file_location("benchmark_tests","/input/tests.py")
tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
def runner(stdin_text):
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        p=subprocess.Popen([sys.executable,"/input/program.py"],stdin=subprocess.PIPE,stdout=out,stderr=err,text=True)
        try:
            p.communicate(input=stdin_text,timeout=20)
        except BaseException:
            p.kill();p.wait();raise
        out.seek(0);err.seek(0)
        stdout=out.read(4194305);stderr=err.read(4194305)
        if len(stdout)>4194304 or len(stderr)>4194304: raise RuntimeError("bounded output exceeded")
        if p.returncode: print(stderr.decode(errors="replace")[-4096:],file=sys.stderr)
        return stdout.decode(),p.returncode
tests.test_cases(runner)
print("REFERENCE_TESTS_PASS")
'''


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sandbox_command(task, environment):
    environment = environment.absolute()
    python_tree = Path(sys.executable).resolve().parent.parent
    interpreter = environment / "bin/python"
    # uv's venv symlink can point through a second version-alias symlink.
    # Mount ONLY that interpreter tree at both names, not the user's uv cache.
    linked = interpreter.readlink() if interpreter.is_symlink() else interpreter
    linked = linked if linked.is_absolute() else interpreter.parent / linked
    python_alias_tree = linked.parent.parent
    args = ["timeout", "--kill-after=5s", "180s", "prlimit", "--cpu=150", "--as=8589934592", "--core=0", "--fsize=16777216",
            "bwrap", "--clearenv", "--die-with-parent", "--new-session", "--unshare-all", "--cap-drop", "ALL"]
    for directory in ("/usr", "/lib", "/lib64"):
        args += ["--ro-bind", directory, directory]
    args += ["--ro-bind", str(python_tree), str(python_tree),
             "--ro-bind", str(environment), str(environment), "--ro-bind", str(task), "/input",
             "--proc", "/proc", "--dev", "/dev", "--size", "268435456", "--tmpfs", "/tmp",
             "--size", "268435456", "--tmpfs", "/work", "--setenv", "HOME", "/work",
             "--setenv", "PATH", "/usr/bin:/bin", "--setenv", "LANG", "C.UTF-8",
             "--setenv", "PYTHONDONTWRITEBYTECODE", "1", "--setenv", "MPLBACKEND", "Agg",
             "--setenv", "OPENBLAS_NUM_THREADS", "1", "--setenv", "OMP_NUM_THREADS", "1",
             "--chdir", "/work", str(environment / "bin/python"), "/input/driver.py"]
    if python_alias_tree != python_tree:
        args[-2:-2] = ["--ro-bind", str(python_tree), str(python_alias_tree)]
    return args


def qualify(snapshot, environment, destination):
    if destination.exists():
        raise FileExistsError("Preserve qualification output")
    p = json.loads((snapshot / "manifest.json").read_text())
    for name, expected in p["files_sha256"].items():
        if Path(name).name != name or digest(snapshot / name) != expected:
            raise ValueError("Snapshot integrity")
    records = [json.loads(line) for line in (snapshot / "PRIVATE_selected_reference_tests.jsonl").read_text().splitlines()]
    if [r["task_id"] for r in records] != p["selected_ids"]:
        raise ValueError("Cohort drift")
    rows = []
    report = {"complete": False, "scope": "Original external Python reference self-tests ONLY, not Julia/model scores",
              "snapshot_manifest_sha256": digest(snapshot / "manifest.json"), "source_sha256": digest(Path(__file__)),
              "python": sys.version, "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}, "rows": rows}
    def save():
        destination.write_text(json.dumps(report, indent=2) + "\n")
    save()
    for record in records:
        with tempfile.TemporaryDirectory(prefix="bcbx-reference-") as root:
            task = Path(root)
            (task / "program.py").write_text(record["program"])
            (task / "tests.py").write_text(record["test_suite"])
            (task / "driver.py").write_text(DRIVER)
            started = time.monotonic()
            result = subprocess.run(sandbox_command(task, environment), capture_output=True, text=True, timeout=190)
            passed = result.returncode == 0 and result.stdout.strip() == "REFERENCE_TESTS_PASS"
            rows.append({"task_id": record["task_id"], "passed": passed, "returncode": result.returncode,
                         "seconds": time.monotonic() - started, "stdout_tail": result.stdout[-4096:], "stderr_tail": result.stderr[-4096:]})
            save()
            print(record["task_id"], "PASS" if passed else "FAIL", flush=True)
    report["complete"] = True
    report["all_reference_tests_passed"] = all(r["passed"] for r in rows)
    save()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("environment", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    qualify(args.snapshot, args.environment, args.destination)
