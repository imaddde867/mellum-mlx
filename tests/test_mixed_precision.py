import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import convert


class MixedPrecisionTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(callable(getattr(convert, 'mixed_plan', None)), 'mixed policy missing')
        self.assertTrue(callable(getattr(convert, 'estimate_payload', None)), 'size estimator missing')
        self.assertTrue(callable(getattr(convert, 'realized_map', None)), 'receipt verification missing')

    def shapes(self, tied=False):
        shapes = {'model.embed_tokens': (128, 64)}
        if not tied:
            shapes['lm_head'] = (128, 64)
        for name in ['q', 'k', 'v', 'o']:
            shapes[f'model.layers.0.self_attn.{name}_proj'] = (64, 64)
        shapes['model.layers.0.mlp.gate'] = (8, 64)
        for name in ['gate', 'up', 'down']:
            shapes[f'model.layers.0.mlp.switch_mlp.{name}_proj'] = (8, 64, 64)
        return shapes

    def test_complete_recipes_keep_experts_four_and_routers_eight(self):
        for recipe, attention, embedding in [('A', 6, 4), ('B', 4, 6), ('C', 6, 6)]:
            plan = convert.mixed_plan(self.shapes(), {'num_hidden_layers': 1, 'tie_word_embeddings': False}, recipe)
            for name in ['q', 'k', 'v', 'o']:
                self.assertEqual(plan[f'model.layers.0.self_attn.{name}_proj']['bits'], attention)
            self.assertEqual(plan['model.embed_tokens']['bits'], embedding)
            self.assertEqual(plan['lm_head']['bits'], embedding)
            self.assertEqual(plan['model.layers.0.mlp.gate']['bits'], 8)
            for name in ['gate', 'up', 'down']:
                self.assertEqual(plan[f'model.layers.0.mlp.switch_mlp.{name}_proj']['bits'], 4)
            self.assertNotIn('model.norm', plan)

    def test_tied_head_uses_only_embedding_and_missing_coverage_blocks(self):
        cfg = {'num_hidden_layers': 1, 'tie_word_embeddings': True}
        plan = convert.mixed_plan(self.shapes(True), cfg, 'B')
        self.assertEqual(plan['model.embed_tokens']['bits'], 6)
        self.assertNotIn('lm_head', plan)
        shapes = self.shapes(True)
        del shapes['model.layers.0.self_attn.k_proj']
        with self.assertRaises(ValueError):
            convert.mixed_plan(shapes, cfg, 'A')
        shapes = self.shapes(True)
        shapes['lm_head'] = (128, 64)
        with self.assertRaises(ValueError):
            convert.mixed_plan(shapes, cfg, 'B')

    def test_size_includes_bf16_scales_biases_and_excluded_tensors(self):
        tensors = {'x.weight': {'shape': [2, 64], 'nbytes': 256},
                   'norm.weight': {'shape': [64], 'nbytes': 128}}
        plan = {'x': {'bits': 6, 'group_size': 64, 'mode': 'affine'}}
        self.assertEqual(convert.estimate_payload(tensors, plan), 232)

    def test_d_changes_only_expert_down_groups(self):
        cfg = {'num_hidden_layers': 1, 'tie_word_embeddings': False}
        c = convert.mixed_plan(self.shapes(), cfg, 'C')
        d = convert.mixed_plan(self.shapes(), cfg, 'D')
        for path, policy in d.items():
            expected = dict(c[path])
            if path.endswith('.switch_mlp.down_proj'):
                expected['group_size'] = 32
            self.assertEqual(policy, expected)

    def test_group32_estimate_adds_real_mellum_down_overhead(self):
        tensors = {'down.weight': {'shape': [28, 64, 2304, 896], 'nbytes': 7398752256}}
        c = {'down': {'bits': 4, 'group_size': 64, 'mode': 'affine'}}
        d = {'down': {'bits': 4, 'group_size': 32, 'mode': 'affine'}}
        self.assertEqual(convert.estimate_payload(tensors, d) - convert.estimate_payload(tensors, c),
                         231211008)

    def test_receipt_rejects_silently_changed_precision(self):
        plan = {'x': {'bits': 6, 'group_size': 64, 'mode': 'affine'}}
        self.assertEqual(convert.realized_map({'bits': 4, 'group_size': 64, 'mode': 'affine', **plan}, plan), plan)
        with self.assertRaises(ValueError):
            convert.realized_map({'bits': 4, 'group_size': 64, 'mode': 'affine'}, plan)
