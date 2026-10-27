import copy
import unittest
from reference_bayes_greedy_cpu import validate_requests


class PublicPacketTests(unittest.TestCase):
    def setUp(self):
        self.packet = {"jobs": [{"id": f"bayes_extract_0{i}_direct_probability", "task": str(i),
            "messages": [{"role": "system", "content": "Public schema"}, {"role": "user", "content": "Public problem"}],
            "input_ids": [1, 2, 3]} for i in range(1, 9)]}

    def test_all_eight(self):
        self.assertEqual(len(validate_requests(self.packet)), 8)

    def test_no_answer_fields(self):
        p = copy.deepcopy(self.packet)
        p["jobs"][0]["expected_probability"] = 0.25
        with self.assertRaises(ValueError):
            validate_requests(p)

    def test_duplicate_or_subset_rejected(self):
        for jobs in (self.packet["jobs"][:7], [self.packet["jobs"][0]] * 8):
            with self.assertRaises(ValueError):
                validate_requests({"jobs": jobs})

    def test_bad_token_ids(self):
        p = copy.deepcopy(self.packet)
        p["jobs"][0]["input_ids"] = [True]
        with self.assertRaises(ValueError):
            validate_requests(p)


if __name__ == "__main__":
    unittest.main()
