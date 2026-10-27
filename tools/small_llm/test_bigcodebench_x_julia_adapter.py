import json
import hashlib
from pathlib import Path
import sys
import unittest

from bigcodebench_x_julia_adapter import evaluate_julia


class JuliaAdapterTests(unittest.TestCase):
    def test_real_julia_stdin_stdout_and_negative_control(self):
        environment = Path(sys.executable).absolute().parent.parent
        tests = "def test_cases(run):\n    assert run('2 3\\n')==('5\\n',0)\n    assert run('-7 4\\n')==('-3\\n',0)\n"
        correct = "values=parse.(Int,split(read(stdin,String)));println(sum(values))"
        incorrect = "values=parse.(Int,split(read(stdin,String)));println(prod(values))"
        positive = evaluate_julia(correct, tests, environment)
        negative = evaluate_julia(incorrect, tests, environment)
        self.assertTrue(positive["passed"], positive)
        self.assertFalse(negative["passed"])
        self.assertEqual(positive["test_suite_sha256"], negative["test_suite_sha256"])
        self.assertNotEqual(positive["program_sha256"], negative["program_sha256"])

    def test_source_bounds(self):
        with self.assertRaises(ValueError):
            evaluate_julia("x" * 262145, "pass", Path("/unused"))
        with self.assertRaises(TypeError):
            evaluate_julia(None, "pass", Path("/unused"))

    def test_real_headless_makie_png_contract(self):
        environment = Path(sys.executable).absolute().parent.parent
        program = '''
using CairoMakie, Base64
CairoMakie.activate!(px_per_unit=1)
f=Figure(size=(160,120));ax=Axis(f[1,1]);lines!(ax,[1,2,3],[1,4,2])
io=IOBuffer();show(io,MIME"image/png"(),f);println(base64encode(take!(io)))
'''
        tests = '''
def test_cases(run):
    import base64, io
    from PIL import Image
    output,code=run('')
    assert code==0
    image=Image.open(io.BytesIO(base64.b64decode(output.strip())))
    assert image.size==(160,120)
    assert image.format=='PNG'
'''
        result = evaluate_julia(program, tests, environment)
        self.assertTrue(result["passed"], result)


if __name__ == "__main__":
    result = unittest.main(exit=False).result
    directory=Path(__file__).parent
    report={"complete":result.wasSuccessful(),"tests":result.testsRun,
            "scope":"Trusted Julia stdin/stdout, negative-control and headless PNG adapter qualification only; NOT model performance",
            "source_sha256":{name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in
                ("bigcodebench_x_julia_adapter.py","qualify_bigcodebench_x_reference.py","test_bigcodebench_x_julia_adapter.py")}}
    (directory/"bigcodebench_x_julia_adapter_validation.json").write_text(json.dumps(report,indent=2)+"\n")
    sys.exit(0 if result.wasSuccessful() else 1)
