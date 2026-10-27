import copy
import unittest
import capture_julia_docs_v2 as module


class ExpandedDocsTests(unittest.TestCase):
    def fixture(self):
        return {"modules": list(module.MODULES), "gpu_nodes_visible": False,
                "bindings": {m: ["example"] for m in module.MODULES},
                "rows": [{"qualified": "Base.example", "text": "public docs", "binding_id": 1}]}

    def test_valid_source_only_fixture(self):
        self.assertIsNone(module.validate(self.fixture()))

    def test_trusted_capture_does_not_modify_candidate_execution(self):
        original = list(module.JULIA_COMMAND)
        command = module.trusted_capture_command()
        self.assertIn("--compile=yes", command)
        self.assertIn("-O1", command)
        self.assertNotIn("--compile=min", command)
        self.assertEqual(module.JULIA_COMMAND, original)

    def test_wrong_or_incomplete_inventory_rejected(self):
        for mutation in (lambda p: p.update(gpu_nodes_visible=True), lambda p: p["modules"].pop(),
                         lambda p: p["bindings"].pop("Dates"), lambda p: p["rows"].append(dict(p["rows"][0])),
                         lambda p: p["rows"][0].update(binding_id=True), lambda p: p["rows"][0].update(binding_id=0),
                         lambda p: p["rows"][0].update(text="x" * 8001),
                         lambda p: p["rows"][0].update(qualified="Base.absent")):
            with self.subTest(mutation=mutation):
                packet = copy.deepcopy(self.fixture())
                mutation(packet)
                with self.assertRaises(ValueError):
                    module.validate(packet)


if __name__ == "__main__":
    unittest.main()
