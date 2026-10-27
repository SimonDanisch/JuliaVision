import unittest
from capture_julia_binding_identity import attach_bindings


class BindingIdentityTests(unittest.TestCase):
    def test_preserve_text_and_distinguish_same_text_different_objects(self):
        docs = dict(complete=True, julia_version="test", rows=[
            dict(qualified="A.f", text="shared text"), dict(qualified="B.f", text="shared text"),
            dict(qualified="A.g", text="shared text")])
        captured = dict(julia_version="test", rows=[
            dict(qualified="A.f", binding_id=1), dict(qualified="B.f", binding_id=1),
            dict(qualified="A.g", binding_id=2)])
        rows = attach_bindings(docs, captured)
        self.assertEqual([r["text"] for r in rows], [r["text"] for r in docs["rows"]])
        self.assertEqual([r["binding_id"] for r in rows], [1, 1, 2])
        self.assertNotIn("binding_id", docs["rows"][0])

    def test_reject_bad_identity_cohort_version(self):
        docs = dict(complete=True, julia_version="test", rows=[dict(qualified="A.f", text="doc")])
        for captured in (
            dict(julia_version="other", rows=[dict(qualified="A.f", binding_id=1)]),
            dict(julia_version="test", rows=[dict(qualified="A.g", binding_id=1)]),
            dict(julia_version="test", rows=[dict(qualified="A.f", binding_id=True)]),
            dict(julia_version="test", rows=[dict(qualified="A.f", binding_id=0)]),
            dict(julia_version="test", rows=[]),
        ):
            with self.assertRaises(ValueError): attach_bindings(docs, captured)


if __name__ == "__main__":
    unittest.main()
