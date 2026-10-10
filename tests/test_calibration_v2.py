import copy
import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


class CalibrationV2Tests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('calibration_v2'), 'corpus preparation missing')
        import calibration_v2
        return calibration_v2

    def test_family_leakage_is_rejected_before_variants(self):
        m = self.module()
        m.check_splits({'train': ['a'], 'valid': ['b'], 'reserved': ['c']})
        with self.assertRaises(ValueError):
            m.check_splits({'train': ['a'], 'valid': ['a'], 'reserved': ['c']})

    def test_tool_schema_rejects_unknown_missing_and_wrong_type(self):
        m = self.module()
        m.check_call({'name': 'edit_file', 'arguments': {'path': 'task.py', 'content': 's="quoted"\n'}})
        for args in [{'path': 'task.py'}, {'path': 'task.py', 'content': 1},
                     {'path': 'task.py', 'content': '', 'extra': True}]:
            with self.assertRaises(ValueError):
                m.check_call({'name': 'edit_file', 'arguments': args})
        with self.assertRaises(ValueError):
            m.check_call({'name': 'execute_shell', 'arguments': {'command': 'anything'}})

    def test_incomplete_or_rewritten_trajectories_are_rejected(self):
        m = self.module()
        row = {'id': 'a', 'family': 'a', 'kind': 'coding', 'origin': 'authored',
               'solution': 'def a():\n    return "quote"\n', 'messages': [
                   {'role': 'user', 'content': 'Implement a.'},
                   {'role': 'assistant', 'reasoning_content': 'Return a literal.',
                    'content': '```python\ndef a():\n    return "quote"\n```'}]}
        m.check_trajectory(row)
        for field, value in [('finish_reason', 'length'), ('solution', 'def a(): return 1')]:
            bad = copy.deepcopy(row); bad[field] = value
            with self.assertRaises(ValueError):
                m.check_trajectory(bad)
        bad = copy.deepcopy(row); bad['messages'].pop()
        with self.assertRaises(ValueError):
            m.check_trajectory(bad)

    def test_region_labels_cover_all_targets_and_transition(self):
        m = self.module()
        text = '<|im_start|>assistant\n<think>\nreason\n</think>\n\nanswer<|im_end|>\n'
        offsets = [(i, i+1) for i in range(len(text))]
        labels = m.regions(text, offsets, transition_radius=0)
        self.assertEqual(len(labels), len(text)-1)
        self.assertEqual(labels[text.index('reason')-1], 'reasoning')
        self.assertEqual(labels[text.index('</think>')-1], 'transition')
        self.assertEqual(labels[text.index('answer')-1], 'final')
        self.assertEqual(labels[0], 'context')

    def test_reserved_cannot_enter_scoring(self):
        m = self.module()
        with self.assertRaises(ValueError):
            m.scoring_rows([{'split': 'reserved', 'id': 'a'}])
        with self.assertRaises(ValueError):
            m.scoring_rows([{'split': 'train', 'id': 'a'}])


class CapturedReductionTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == 'linux' and importlib.util.find_spec('mlx'), 'numerical checks run on pinned CUDA host')
    def test_identical_capture_kl_orientation_and_uniform_aggregate(self):
        import math
        import mlx.core as mx
        mx.set_default_device(mx.cpu)
        import calibration_v2 as m
        self.assertTrue(callable(getattr(m, 'reduce_captured', None)), 'captured-logit reducer missing')
        teacher = mx.array([[[2*math.log(.75),2*math.log(.25)]]*2], dtype=mx.float32)
        student = mx.zeros_like(teacher)
        result = m.reduce_captured(student, teacher, ['reasoning','final'])
        expected = .75*math.log(1.5)+.25*math.log(.5)
        self.assertAlmostEqual(result['fp32']['aggregate'][0]/2, expected, places=6)
        self.assertEqual(result['fp32']['aggregate'][1],2)
        self.assertAlmostEqual(result['fp32']['reasoning'][0], expected, places=6)
        identical=m.reduce_captured(teacher,teacher,['reasoning','final'])
        self.assertAlmostEqual(identical['fp32']['aggregate'][0],0.,places=6)

class PinnedRenderingTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == 'linux' and Path('work/source/tokenizer.json').exists(), 'pinned tokenizer checks run on CUDA host')
    def test_structured_tokenizer_output_preserves_full_tool_conversation(self):
        import json
        import calibration_v2 as m
        from transformers import AutoTokenizer
        self.assertTrue(callable(getattr(m, 'render_trajectory', None)), 'pinned renderer missing')
        tokenizer=AutoTokenizer.from_pretrained('work/source')
        family=json.loads(Path('fixtures/calibration-v2/families.json').read_text())[0]
        row=list(m.trajectories(family, {'passed':True}))[3]
        text, tokens, labels=m.render_trajectory(tokenizer,row)
        self.assertGreater(len(tokens),256)
        self.assertIn(row['solution'],text)
        self.assertEqual(len(labels),len(tokens)-1)
        self.assertTrue(all(region in labels for region in ['reasoning','transition','final','tool_call']))
        self.assertEqual(tokens,tokenizer.encode(text,add_special_tokens=False))



class BaselineSummaryTests(unittest.TestCase):
    def receipt(self, values, context=1.):
        return {'status':'complete', 'updates':0, 'data_manifest_sha256':'same',
            'protocol_sha256':'same', 'repeats':[{'totals':{mode:{
                region:{'mean':context if region=='context' else v, 'tokens':10}
                for region in ('context','reasoning','transition','final','tool_call','aggregate')}
                for mode in ('legacy','fp32')}} for v in values]}

    def test_gap_requires_reproducible_assistant_loss_not_just_context(self):
        import calibration_v2 as m
        self.assertTrue(callable(getattr(m, 'summarize', None)), 'baseline summary missing')
        result=m.summarize({'C':self.receipt([.2]*5),'mxfp4':self.receipt([.1]*5)})
        self.assertTrue(result['relevant_gap'])
        self.assertTrue(result['C_trails_mxfp4'])
        result=m.summarize({'C':self.receipt([.001]*5),'mxfp4':self.receipt([.001]*5)})
        self.assertFalse(result['relevant_gap'])
        result=m.summarize({'C':self.receipt([.001,.001,.001,.001,.3]),'mxfp4':self.receipt([.1]*5)})
        self.assertFalse(result['relevant_gap'])

    def test_incomplete_or_different_teacher_contexts_cannot_support_a_gap(self):
        import calibration_v2 as m
        self.assertTrue(callable(getattr(m, 'summarize', None)), 'baseline summary missing')
        first=self.receipt([.2]*5)
        for bad in [self.receipt([.1]*4),{**first,'updates':1},
                    {**first,'data_manifest_sha256':'different'}]:
            with self.assertRaises(ValueError):
                m.summarize({'C':first,'mxfp4':bad})


if __name__ == '__main__':
    unittest.main()
