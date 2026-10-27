import copy
import unittest
from snapshot_pubmedqa_pilot import partition


class PublicBiomedicalSnapshot(unittest.TestCase):
    def setUp(self):
        self.data = {str(i): {"QUESTION": f"Research question {i}?", "CONTEXTS": [f"Method {i}", f"Result {i}"],
                              "final_decision": ("yes", "no", "maybe")[i % 3], "LONG_ANSWER": "PRIVATE_CONCLUSION",
                              "reasoning_free_pred": "PRIVATE_PREDICTION"} for i in range(1, 31)}

    def test_fixed_hash_and_no_answer_fields(self):
        public, gold = partition(self.data)
        self.assertEqual(len(public), 16)
        self.assertEqual({r["pmid"] for r in public}, set(gold))
        for row in public:
            self.assertEqual(set(row), {"pmid", "question", "abstract_context"})
            self.assertNotIn("PRIVATE_CONCLUSION", str(row))
            self.assertNotIn("PRIVATE_PREDICTION", str(row))
        self.assertEqual(partition(dict(reversed(list(self.data.items())))), (public, gold))

    def test_no_label_or_context_length_selection(self):
        public, _ = partition(self.data)
        changed = copy.deepcopy(self.data)
        for row in changed.values():
            row["final_decision"] = "maybe"
            row["CONTEXTS"] *= 20
        public2, _ = partition(changed)
        self.assertEqual([r["pmid"] for r in public], [r["pmid"] for r in public2])

    def test_malformed_population_refuses_not_filters(self):
        for field, value in (("QUESTION", ""), ("CONTEXTS", []), ("CONTEXTS", [""]),
                             ("final_decision", "unknown")):
            changed = copy.deepcopy(self.data)
            changed["1"][field] = value
            with self.assertRaises(ValueError):
                partition(changed)
        for count in (0, 31, True):
            with self.assertRaises(ValueError):
                partition(self.data, count)


if __name__ == "__main__":
    unittest.main()
