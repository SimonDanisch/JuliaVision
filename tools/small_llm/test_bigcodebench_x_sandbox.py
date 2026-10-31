import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("qualifier", Path(__file__).with_name("qualify_bigcodebench_x_reference.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SandboxTests(unittest.TestCase):
    def test_actual_cpu_jail(self):
        environment = Path(sys.executable).absolute().parent.parent
        with tempfile.TemporaryDirectory() as root:
            outer = Path(root)
            secret = outer / "host_secret.txt"
            secret.write_text("MUST_NOT_BE_VISIBLE")
            task = outer / "input"
            task.mkdir()
            code = f'''
import os, pathlib, socket
assert not os.path.exists({str(secret)!r})
assert not os.path.exists('/dev/dri')
assert not os.path.exists('/home/sim/Programmieren/VulkanDev/Project.toml')
try:
    pathlib.Path('/input/forbidden_write').write_text('bad')
except OSError:
    pass
else:
    raise AssertionError('input not readonly')
connection=socket.socket();connection.settimeout(.2)
try:
    connection.connect(('1.1.1.1',443))
except OSError:
    pass
else:
    raise AssertionError('network available')
finally:
    connection.close()
print('cpu-only-jail-pass')
'''
            (task / "program.py").write_text(code)
            (task / "tests.py").write_text("def test_cases(run):\n    assert run('')==('cpu-only-jail-pass\\n',0)\n")
            (task / "driver.py").write_text(module.DRIVER)
            result = subprocess.run(module.sandbox_command(task, environment), capture_output=True, text=True, timeout=190)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "REFERENCE_TESTS_PASS")
            self.assertEqual(secret.read_text(), "MUST_NOT_BE_VISIBLE")
            self.assertFalse((task / "forbidden_write").exists())

    def test_existing_receipt_never_overwritten(self):
        with tempfile.TemporaryDirectory() as root:
            receipt = Path(root) / "receipt.json"
            receipt.write_text("preserved")
            with self.assertRaises(FileExistsError):
                module.qualify(Path(root), Path(root), receipt)
            self.assertEqual(receipt.read_text(), "preserved")


if __name__ == "__main__":
    unittest.main()
