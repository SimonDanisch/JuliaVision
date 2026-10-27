import copy
import unittest
from snapshot_pubmedqa_pilot import partition
from snapshot_pubmedqa_retrieval import public_corpus, next_cohort


class PublicRetrievalSnapshot(unittest.TestCase):
    def setUp(self):
        self.data = {str(i): {"QUESTION": "PRIVATE_TITLE", "CONTEXTS": ["PUBLIC_BODY", str(i)],
                              "final_decision": ("yes", "no", "maybe")[i % 3], "LONG_ANSWER": "PRIVATE_CONCLUSION",
                              "reasoning_free_pred": "PRIVATE_PREDICTION"} for i in range(1, 65)}
        self.first = [r["pmid"] for r in partition(self.data)[0]]

    def test_disjoint_next_fixed_hash_cohort(self):
        tasks, gold = next_cohort(self.data, self.first)
        self.assertEqual(len(tasks), 16)
        self.assertFalse(set(self.first) & set(gold))
        self.assertEqual(tasks, partition(self.data, 32)[0][16:])
        with self.assertRaises(ValueError):
            next_cohort(self.data, self.first[:-1])

    def test_corpus_has_no_titles_conclusions_or_labels(self):
        rows = public_corpus(self.data)
        self.assertEqual(len(rows), 64)
        self.assertTrue(all(set(row) == {"qualified", "text"} for row in rows))
        self.assertNotIn("PRIVATE_", str(rows))
        changed = copy.deepcopy(self.data)
        for row in changed.values():
            row["final_decision"] = "maybe"
        self.assertEqual(public_corpus(changed), rows)
        self.assertEqual([r["pmid"] for r in next_cohort(changed, self.first)[0]],
                         [r["pmid"] for r in next_cohort(self.data, self.first)[0]])


if __name__ == "__main__":
    unittest.main()
