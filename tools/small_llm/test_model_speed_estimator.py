import json
import math
import unittest
from pathlib import Path
import model_speed_estimator as M


class SpeedEstimatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models={r["repo"]:r["config"] for r in json.loads((M.HERE/"model_speed_metadata.json").read_text())}

    def test_quantization_metadata(self):
        self.assertEqual(M.matbytes(256,256),256*144)
        self.assertEqual(M.matbytes(256,256,head=True),256*210)
        self.assertEqual(M.matbytes(32,256),32*256*4)
        with self.assertRaises(ValueError): M.matbytes(256,257)

    def test_real_profile_calibration(self):
        predicted=M.qwen_traffic(self.models["Qwen/Qwen3.5-4B"])
        measured=json.loads((M.HERE/"qwen4b_decode_profile.json").read_text())["inventory"]["nominal_weight_read_bytes"]
        self.assertLess(abs(predicted["active_weight_bytes"]-measured)/measured,0.001)
        self.assertEqual(predicted["kv_bytes_per_position_f16"],32768)
        # F32 input embeddings occupy memory but are NOT streamed in full.
        self.assertGreater(predicted["resident_text_weight_bytes"],predicted["active_weight_bytes"]+2e9)

    def test_moe_capacity_is_not_active_traffic(self):
        p=M.qwen_traffic(self.models["Qwen/Qwen3.6-35B-A3B"])
        self.assertGreater(p["resident_text_weight_bytes"],10*p["active_weight_bytes"])
        self.assertLess(p["active_weight_bytes"],M.qwen_traffic(self.models["Qwen/Qwen3.5-4B"])["active_weight_bytes"])
        self.assertEqual(p["kv_bytes_per_position_f16"],20480)

    def test_gpt_oss_formats_and_window(self):
        c=self.models["openai/gpt-oss-20b"]
        native=M.gpt_oss_traffic(c,True);requant=M.gpt_oss_traffic(c,False)
        self.assertGreater(native["active_weight_bytes"],requant["active_weight_bytes"])
        self.assertAlmostEqual(native["resident_text_weight_bytes"]/2**30,12.8,delta=.3)
        self.assertEqual(native["kv_bytes_per_position_f16"],24576)
        self.assertEqual(native["local_kv_fixed_bytes_f16"],24576*128)

    def test_speed_monotonicity_and_bounds(self):
        self.assertLess(M.speed(3e9,0)[1],M.speed(2e9,0)[1])
        self.assertLess(M.speed(2e9,1e9)[1],M.speed(2e9,0)[1])
        low,high=M.speed(2e9,1e9,extra_ms=(1,4))
        self.assertLessEqual(low,high)
        self.assertLess(high,256e9/3e9)
        for args in [(0,0),(-1,0),(1,-1)]:
            with self.assertRaises(ValueError): M.speed(*args)

    def test_held_out_2b_calibration_sanity(self):
        p=M.qwen_traffic(self.models["Qwen/Qwen3.5-2B"])
        actual=json.loads((M.HERE/"qwen2b_q4_resident_benchmark.json").read_text())["median_decode_tps"]
        low,high=M.speed(p["active_weight_bytes"],0)
        self.assertLessEqual(low,actual);self.assertLessEqual(actual,high)

    def test_unknown_architecture_fails_closed(self):
        with self.assertRaises(ValueError): M.qwen_traffic(self.models["Qwen/Qwen3.8-Flash-Next"])
        coarse=M.advertised_traffic(4,26,self.models["google/gemma-4-26B-A4B-it"],"coarse")
        self.assertTrue(all(r["scenario_tps"] is None for r in M.scenarios(coarse)["contexts"]))

    def test_implicit_hybrid_layer_layout(self):
        c=self.models["Qwen/Qwen3-Coder-Next"]
        p=M.qwen_traffic(c)
        self.assertEqual(p["kv_bytes_per_position_f16"],12*2*2*256*2)
        explicit=dict(c,layer_types=["full_attention" if (i+1)%4==0 else "linear_attention" for i in range(48)])
        self.assertEqual(p,M.qwen_traffic(explicit))
        with self.assertRaises(ValueError): M.qwen_traffic(dict(c,full_attention_interval=0))


if __name__=="__main__": unittest.main()
