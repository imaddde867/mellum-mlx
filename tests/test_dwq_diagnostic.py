import math
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import refine_dwq


class DiagnosticGateTests(unittest.TestCase):
    def test_step_zero_wins_when_loss_reduction_is_within_variability(self):
        self.assertTrue(callable(getattr(refine_dwq, "select_checkpoint", None)), "checkpoint gate missing")
        self.assertEqual(refine_dwq.select_checkpoint([.100, .102, .101],
            {8: [.100, .101, .0999], 16: [.103, .104, .1035]}), 0)

    def test_clear_improvement_selects_best_checkpoint_not_final_step(self):
        self.assertTrue(callable(getattr(refine_dwq, "select_checkpoint", None)), "checkpoint gate missing")
        self.assertEqual(refine_dwq.select_checkpoint([.100, .102, .101],
            {8: [.090, .091, .0905], 16: [.080, .081, .0805],
             24: [.095, .096, .0955], 32: [.110, .111, .1105]}), 16)

    def test_rejects_incomplete_nonfinite_or_unscheduled_measurements(self):
        self.assertTrue(callable(getattr(refine_dwq, "select_checkpoint", None)), "checkpoint gate missing")
        for baseline, checkpoints in [([], {}), ([.1], {8:[.09]}),
            ([.1,.1,.1], {4:[.09,.09,.09]}),
            ([.1,.1,.1], {8:[.09,math.nan,.09]}),
            ([.1,.1,.1], {8:[.09,.09]})]:
            with self.assertRaises(ValueError):
                refine_dwq.select_checkpoint(baseline, checkpoints)

    def test_diagnostic_rejects_degraded_or_unpinned_parent(self):
        self.assertTrue(callable(getattr(refine_dwq, 'require_pristine_c', None)), 'pristine-parent guard missing')
        recipe={'recipe':'C', 'source_revision':refine_dwq.REVISION}
        refine_dwq.require_pristine_c(recipe)
        for bad in [{**recipe,'calibration':{}}, {**recipe,'recipe':'D'},
                    {**recipe,'source_revision':'different'}]:
            with self.assertRaises(ValueError):
                refine_dwq.require_pristine_c(bad)
