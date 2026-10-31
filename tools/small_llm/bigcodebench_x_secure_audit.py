"""Separate posthoc test-blind execution audit; original grader stays frozen."""
import hashlib
from pathlib import Path
import subprocess
import tempfile
import time

from bigcodebench_x_julia_adapter import JULIA_COMMAND, JULIA_DRIVER, julia_sandbox_command

# Nested unprivileged namespaces are unavailable on this machine. Instead the
# trusted test source enters the jailed parent via a pipe, never a mounted file.
# Disable parent dumpability BEFORE loading it, close that pipe, then launch
# candidates with only their example input and closed extra descriptors.
OLD_LOADER='''spec=importlib.util.spec_from_file_location("benchmark_tests","/input/tests.py")
tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)'''
NEW_LOADER='''import ctypes, types
libc=ctypes.CDLL(None)
if libc.prctl(4,0,0,0,0)!=0 or libc.prctl(3,0,0,0,0)!=0:
    raise RuntimeError("test-parent nondumpable gate failed")
source=sys.stdin.read(262145);sys.stdin.close()
if len(source.encode())>262144: raise ValueError("bounded private tests")
tests=types.ModuleType("benchmark_tests")
exec(compile(source,"<private-benchmark-tests>","exec"),tests.__dict__)
source=None'''
if JULIA_DRIVER.count(OLD_LOADER)!=1:
    raise ValueError("Frozen loader drift")
SECURE_DRIVER=JULIA_DRIVER.replace(OLD_LOADER,NEW_LOADER)


def evaluate_secure_julia(program, test_suite, environment):
    if not isinstance(program, str) or not isinstance(test_suite, str):
        raise TypeError("Code/test strings required")
    if len(program.encode())>262144 or len(test_suite.encode())>262144:
        raise ValueError("Bounded sources required")
    with tempfile.TemporaryDirectory(prefix="bcbx-secure-") as root:
        task=Path(root)
        (task/"program.jl").write_text(program)
        (task/"driver.py").write_text(SECURE_DRIVER)
        started=time.monotonic()
        result=subprocess.run(julia_sandbox_command(task,environment),input=test_suite,capture_output=True,text=True,timeout=190)
        return {"passed":result.returncode==0 and result.stdout.strip()=="REFERENCE_TESTS_PASS",
                "returncode":result.returncode,"seconds":time.monotonic()-started,
                "stdout_tail":result.stdout[-4096:],"stderr_tail":result.stderr[-4096:],
                "program_sha256":hashlib.sha256(program.encode()).hexdigest(),
                "test_suite_sha256":hashlib.sha256(test_suite.encode()).hexdigest(),
                "scope":"Separate POSTHOC test-blind execution audit; no regeneration or replacement of registered baseline scores"}
