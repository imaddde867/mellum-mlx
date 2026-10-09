"""Round-trip validation must reject nonfinite and incomplete comparisons."""
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("dwq_pilot", Path(__file__).parents[1] / "scripts/dwq_pilot.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class RoundTripTests(unittest.TestCase):
    def test_accepts_identical_finite_result(self):
        pilot.check_roundtrip(0.0, 0.0, [])

    def test_rejects_nonfinite_reload_and_repeat(self):
        for invalid in [float("nan"), float("inf"), -float("inf")]:
            for delta, repeated in [(invalid, 0.0), (0.0, invalid)]:
                with self.subTest(delta=delta, repeated=repeated):
                    with self.assertRaises(ValueError):
                        pilot.check_roundtrip(delta, repeated, [])

    def test_rejects_changed_outputs_and_parameters(self):
        for delta, repeated, changed in [(0.01, 0.0, []), (0.0, 0.01, []), (0.0, 0.0, ["weight"])]:
            with self.subTest(delta=delta, repeated=repeated, changed=changed):
                with self.assertRaises(ValueError):
                    pilot.check_roundtrip(delta, repeated, changed)
