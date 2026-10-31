import hashlib
import json
from pathlib import Path
import sys
import unittest

from bigcodebench_x_secure_audit import evaluate_secure_julia


class SecureAuditTests(unittest.TestCase):
    def test_candidate_cannot_see_tests_and_arithmetic_still_works(self):
        program='''
@assert !isfile("/input/tests.py")
@assert !occursin("PRIVATE_PAYLOAD_SENTINEL_92187",read("/input/driver.py",String))
@assert !isdir("/dev/dri")
parent=ccall(:getppid,Cint,())
for file in ("/proc/$parent/mem","/proc/$parent/fd/0")
    readable=try;open(io->nothing,file);true;catch;false;end
    @assert !readable
end
println(parse(Int,readline())+1)
'''
        tests="# PRIVATE_PAYLOAD_SENTINEL_92187\ndef test_cases(run):\n    assert run('4\\n')==('5\\n',0)\n    assert run('-2\\n')==('-1\\n',0)\n"
        r=evaluate_secure_julia(program,tests,Path(sys.executable).absolute().parent.parent)
        self.assertTrue(r["passed"],r)

    def test_shared_legitimate_file_fixture(self):
        program='path=strip(readline());println(sum(parse.(Int,split(read(path,String)))))'
        tests='''
def test_cases(run):
    import tempfile,pathlib
    with tempfile.TemporaryDirectory() as root:
        path=pathlib.Path(root)/'data.txt';path.write_text('2 3 -1')
        assert run(str(path)+'\\n')==('4\\n',0)
'''
        r=evaluate_secure_julia(program,tests,Path(sys.executable).absolute().parent.parent)
        self.assertTrue(r["passed"],r)

    def test_reading_hidden_tests_is_rejected(self):
        program='println(read("/input/tests.py",String))'
        tests="def test_cases(run):\n    _,code=run('')\n    assert code==0\n"
        r=evaluate_secure_julia(program,tests,Path(sys.executable).absolute().parent.parent)
        self.assertFalse(r["passed"])
        self.assertIn("/input/tests.py",r["stderr_tail"])
        self.assertNotIn("No permissions to create a new namespace",r["stderr_tail"])

    def test_headless_png_contract_unchanged(self):
        program='''
using CairoMakie,Base64
CairoMakie.activate!(px_per_unit=1)
f=Figure(size=(160,120));ax=Axis(f[1,1]);lines!(ax,[1,2,3],[1,4,2])
io=IOBuffer();show(io,MIME"image/png"(),f);println(base64encode(take!(io)))
'''
        tests='''
def test_cases(run):
    import base64,io
    from PIL import Image
    out,code=run('');assert code==0
    image=Image.open(io.BytesIO(base64.b64decode(out.strip())))
    assert image.size==(160,120) and image.format=='PNG'
'''
        r=evaluate_secure_julia(program,tests,Path(sys.executable).absolute().parent.parent)
        self.assertTrue(r["passed"],r)


if __name__=="__main__":
    result=unittest.main(exit=False).result
    directory=Path(__file__).parent
    report={"complete":result.wasSuccessful(),"tests":result.testsRun,
            "scope":"Private test-blind sandbox qualification, NOT model scores",
            "source_sha256":{name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in
                ("bigcodebench_x_secure_audit.py","test_bigcodebench_x_secure_audit.py")}}
    (directory/"bigcodebench_x_secure_audit_validation.json").write_text(json.dumps(report,indent=2)+"\n")
    sys.exit(0 if result.wasSuccessful() else 1)
