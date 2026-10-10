import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))

class ArtifactGateTests(unittest.TestCase):
    def test_reserved_access_requires_selected_matching_verified_artifact(self):
        import importlib.util
        self.assertIsNotNone(importlib.util.find_spec('calibration_v2_exploratory_artifact'),'selected-artifact gate missing')
        import calibration_v2_exploratory_artifact as m
        d={'status':'complete','arm':'S','step':64,'mode':'exploratory','exploratory_protocol_sha256':'p'}
        r={'status':'complete','fidelity_gate_passed':True,'integrity_passed':True,
           'decision_sha256':'d','exploratory_protocol_sha256':'p','artifact':{'weights_sha256':{'model':'w'}}}
        m.check_reserved_gate(d,r,'d','p',{'model':'w'})
        for bad in [{**r,'fidelity_gate_passed':False},{**r,'integrity_passed':False},
                    {**r,'decision_sha256':'other'},{**r,'status':'failed'}]:
            with self.assertRaises(ValueError):m.check_reserved_gate(d,bad,'d','p',{'model':'w'})
        with self.assertRaises(ValueError):m.check_reserved_gate({**d,'arm':None},r,'d','p',{'model':'w'})
        with self.assertRaises(ValueError):m.check_reserved_gate(d,r,'d','p',{'model':'changed'})
