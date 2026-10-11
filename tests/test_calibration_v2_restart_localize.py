import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))

class LocalizationTests(unittest.TestCase):
    def module(self):
        import importlib.util
        self.assertIsNotNone(importlib.util.find_spec('calibration_v2_restart_localize'), 'localization implementation missing')
        import calibration_v2_restart_localize
        return calibration_v2_restart_localize

    def test_comparison_reports_absolute_error_reference_norm_and_module_contributions(self):
        import numpy as np
        m=self.module()
        reference={'layers.0.attention.scales':np.array([3.,4.],np.float32),
                   'layers.1.mlp.biases':np.array([0.,0.],np.float32)}
        actual={'layers.0.attention.scales':np.array([6.,8.],np.float32),
                'layers.1.mlp.biases':np.array([0.,12.],np.float32)}
        value=m.compare_arrays(actual,reference)
        self.assertEqual(value['reference_l2'],5.)
        self.assertEqual(value['absolute_l2'],13.)
        self.assertEqual(value['max_absolute'],12.)
        self.assertEqual(value['relative_l2'],2.6)
        self.assertFalse(value['exact'])
        self.assertEqual(value['groups'][0]['group'],'layers.1.mlp')
        self.assertEqual(value['groups'][0]['squared_error'],144.)
        self.assertEqual(value['groups'][1]['squared_error'],25.)
        same=m.compare_arrays(reference,reference)
        self.assertTrue(same['exact']);self.assertEqual(same['relative_l2'],0.)
        with self.assertRaises(ValueError):m.compare_arrays(actual,{'wrong':np.zeros(2,np.float32)})

    def test_disk_mean_retains_fp32_repeat_order_and_does_not_mutate_inputs(self):
        import numpy as np
        m=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);paths=[]
            for i,value in enumerate([1e8,1.,-1e8,5.,10.]):
                path=root/f'{i}.npy';np.save(path,np.array([value],np.float32));paths.append({'x':path})
            before=[p['x'].read_bytes() for p in paths]
            mean=m.mean_arrays(paths,root/'mean')
            np.testing.assert_array_equal(np.load(mean['x']),np.array([3.],np.float32))
            self.assertEqual(before,[p['x'].read_bytes() for p in paths])
            with self.assertRaises(ValueError):m.mean_arrays(paths[:4],root/'bad')

    @unittest.skipUnless(__import__('importlib').util.find_spec('mlx'), 'MLX unavailable')
    def test_replay_preserves_shared_starting_state_and_uses_supplied_gradient(self):
        import mlx.core as mx
        import numpy as np
        import calibration_v2_runtime as runtime
        from mlx.utils import tree_map
        m=self.module()
        params={'w':mx.array([2.,-3.])};opt=runtime.adam();opt.init(params)
        params=opt.apply_gradients({'w':mx.array([4.,-2.])},params);mx.eval(params,opt.state)
        saved=tree_map(lambda x:x,opt.state)
        before=m.array_digest(saved['w']['m'])
        grads={'w':mx.array([7.,8.])}
        updated,newopt=m.replay_update(params,saved,grads)
        mx.eval(updated,newopt.state)
        self.assertEqual(m.array_digest(saved['w']['m']),before)
        self.assertEqual(saved['step'].item(),1)
        self.assertEqual(newopt.step.item(),2)
        np.testing.assert_allclose(np.array(newopt.state['w']['m']),[1.06,.62],rtol=1e-6)
        again,againopt=m.replay_update(params,saved,grads);mx.eval(again,againopt.state)
        np.testing.assert_array_equal(np.array(updated['w']),np.array(again['w']))
        self.assertEqual(m.optimizer_config(newopt)['betas'],[.9,.999])
        self.assertTrue(m.optimizer_config(newopt)['bias_correction'])
