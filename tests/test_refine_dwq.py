import sys
import hashlib
import json
import shutil
from tempfile import TemporaryDirectory
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import refine_dwq


class DistillationTests(unittest.TestCase):
    def test_only_low_bit_affine_scales_and_biases_may_change(self):
        plan = {'expert': {'bits':4, 'mode':'affine', 'group_size':32},
                'attention': {'bits':6, 'mode':'affine', 'group_size':64},
                'router': {'bits':8, 'mode':'affine', 'group_size':64}}
        refine_dwq.check_changes(['expert.scales','attention.biases'], plan)
        for key in ['expert.weight','router.scales','norm.weight','expert.unknown']:
            with self.assertRaises(ValueError):
                refine_dwq.check_changes([key], plan)
        with self.assertRaises(ValueError):
            refine_dwq.check_changes([], plan)

    def test_calibration_is_complete_disjoint_and_hash_bound(self):
        manifest, datasets = refine_dwq.calibration(Path('fixtures/refinement-v1'))
        self.assertEqual([len(datasets[x]) for x in ['train','valid']], [32,8])
        self.assertTrue(all(64 <= len(tokens) <= 256 for values in datasets.values() for tokens, _ in values))

    def test_calibration_rejects_corruption_and_cross_split_duplicates(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ['manifest.json','train.jsonl','valid.jsonl']:
                shutil.copy2(Path('fixtures/refinement-v1')/name, root/name)
            with (root/'valid.jsonl').open('a') as f:
                f.write('corruption')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                refine_dwq.calibration(root)
            train = (root/'train.jsonl').read_text().splitlines()
            valid = Path('fixtures/refinement-v1/valid.jsonl').read_text().splitlines()
            valid[0] = train[0]
            (root/'valid.jsonl').write_text('\n'.join(valid)+'\n')
            manifest = json.loads((root/'manifest.json').read_text())
            manifest['splits']['valid']['sha256'] = hashlib.sha256((root/'valid.jsonl').read_bytes()).hexdigest()
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                refine_dwq.calibration(root)
