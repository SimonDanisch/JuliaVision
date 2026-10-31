"""Run in the IndexTTS export environment: python -m unittest test_indextts25_export."""

import unittest

from export_indextts25 import prune_unused_weights


class PruneWeightsTests(unittest.TestCase):
    def test_nested_weight_views_survive(self):
        graph = dict(inputs=["x"], outputs=["out"], ops=[{"in": ["second_view", "x"]}],
                     buffers=[dict(id="weight", kind="weight", key="linear.weight"),
                              dict(id="mask", kind="weight", key="attn.bias"),
                              dict(id="first_view", kind="view", of="weight"),
                              dict(id="second_view", kind="view", of="first_view")])
        result = prune_unused_weights(graph)
        self.assertEqual([b["id"] for b in result["buffers"]],
                         ["weight", "first_view", "second_view"])

    def test_interface_weights_survive_without_ops(self):
        graph = dict(inputs=["input"], outputs=["output_view"], ops=[], buffers=[
            dict(id="input", kind="weight", key="input"),
            dict(id="output", kind="weight", key="output"),
            dict(id="output_view", kind="view", of="output"),
            dict(id="unused", kind="weight", key="unused")])
        self.assertEqual([b["id"] for b in prune_unused_weights(graph)["buffers"]],
                         ["input", "output", "output_view"])


if __name__ == "__main__":
    unittest.main()
