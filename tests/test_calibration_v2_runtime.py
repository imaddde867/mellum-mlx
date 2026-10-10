import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))

class RuntimeTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('calibration_v2_runtime'),'runtime runner missing')
        import calibration_v2_runtime
        return calibration_v2_runtime

    def test_runtime_requires_exact_authorized_environment_before_import(self):
        m=self.module()
        expected={'MLX_USE_CUDA_GRAPHS':'1','MLX_CUDA_GRAPH_CACHE_SIZE':'400','MLX_ENABLE_CACHE_THRASHING_CHECK':'0'}
        with patch.dict(os.environ,expected,clear=True):
            actual=m.require_runtime()
            for key,value in expected.items():self.assertEqual(actual[key],value)
            with patch.dict(sys.modules,{'mlx.core':object()}):
                with self.assertRaises(ValueError):m.require_runtime(before_import=True)
        for change in [{'MLX_USE_CUDA_GRAPHS':'0'},{'MLX_CUDA_GRAPH_CACHE_SIZE':'401'},
                       {'MLX_ENABLE_CACHE_THRASHING_CHECK':'1'}]:
            with patch.dict(os.environ,{**expected,**change},clear=True):
                with self.assertRaises(ValueError):m.require_runtime()
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaises(ValueError):m.require_runtime()

    def test_replacement_selection_rejects_old_or_mismatched_bindings(self):
        m=self.module()
        refs={'runtime':'r','protocol':'p','data':'d','targets':'t','validation_targets':'v','source':'s','order':'o'}
        arms={a:{'status':'complete','updates':128,'arm':a,'bindings':refs,'qualification_sha256':'q',
                 'update_order':['first','second']} for a in ['U','S']}
        m.check_replacement(arms,refs,'q',['first','second'])
        arms['U']['bindings']={**refs,'runtime':'old'}
        with self.assertRaises(ValueError):m.check_replacement(arms,refs,'q',['first','second'])
        del arms['U']['bindings']
        with self.assertRaises(ValueError):m.check_replacement(arms,refs,'q',['first','second'])

    def test_selection_refuses_historical_paths_before_reading_arms(self):
        import argparse,json
        m=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            amendment=Path(tmp)/'amendment.json'
            amendment.write_text(json.dumps({'replacement_arms':{'U':'new-U','S':'new-S'}}))
            with patch.object(m,'AMENDMENT',amendment):
                with self.assertRaisesRegex(ValueError,'Selection path'):
                    m.select(argparse.Namespace(u=Path('work/calibration-v2-objective-U'),
                        s=Path('work/calibration-v2-objective-S'),output=Path(tmp)/'decision.json'))
            self.assertFalse((Path(tmp)/'decision.json').exists())

    def test_timeout_terminates_child_and_preserves_log(self):
        import calibration_v2_runtime_process as launcher
        with tempfile.TemporaryDirectory() as tmp:
            log=Path(tmp)/'probe.log'
            result=launcher.run([sys.executable,'-c','import time;print("started",flush=True);time.sleep(60)'],log,.2)
            self.assertTrue(result['timed_out'])
            self.assertIsNone(result['exit_code'])
            self.assertIn('started',log.read_text())
            with self.assertRaises(FileExistsError):launcher.run([sys.executable,'-c','pass'],log,.2)

    def test_training_requires_matching_successful_qualification(self):
        m=self.module()
        self.assertTrue(callable(getattr(m,'check_qualification',None)))
        expected={'runtime':'new','data':'fixed'}
        runtime={'MLX_USE_CUDA_GRAPHS':'1'}
        good={'status':'complete','bindings':expected,'runtime':runtime,'updates':0}
        m.check_qualification(good,expected,runtime)
        for bad in [{**good,'status':'failed'},{**good,'bindings':{'runtime':'old'}},
                    {**good,'runtime':{}},{**good,'updates':1}]:
            with self.assertRaises(ValueError):m.check_qualification(bad,expected,runtime)

    def test_correctness_failure_is_recorded_without_cuda_recovery(self):
        import argparse,json
        m=self.module()
        self.assertTrue(callable(getattr(m,'correctness',None)),'failure receipt handler missing')
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'correctness'
            report={'runtime':{'MLX_USE_CUDA_GRAPHS':'0'},'updates':0}
            # The expensive backend boundary fails; filesystem/error handling stays real.
            with patch.object(m.base,'preflight',side_effect=RuntimeError('cudaMallocAsync out of memory')):
                with self.assertRaisesRegex(RuntimeError,'cudaMallocAsync'):
                    m.correctness(argparse.Namespace(output=output),report)
            saved=json.loads((output/'receipt.json').read_text())
            self.assertEqual(saved['status'],'failed')
            self.assertIn('cudaMallocAsync',saved['error'])
            self.assertEqual(saved['runtime']['MLX_USE_CUDA_GRAPHS'],'0')
            self.assertEqual(saved['updates'],0)
            self.assertFalse(list(output.rglob('*.safetensors')))

@unittest.skipUnless(sys.platform=='linux' and importlib.util.find_spec('mlx'),'restart arrays tested on pinned host')
class RestartTests(unittest.TestCase):
    def test_restart_rejects_nonfinite_master_before_publication(self):
        import mlx.core as mx
        import calibration_v2_runtime as m
        mx.set_default_device(mx.cpu)
        opt=m.adam();params={'weight':mx.array([float('inf')],dtype=mx.float32)}
        opt.init(params);mx.eval(opt.state)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'Nonfinite'):
                m.save_restart(Path(tmp),params,opt,{'arm':'U','cursor':0,'bindings':{}})
            self.assertFalse(list(Path(tmp).glob('step-*')))

    def test_atomic_restart_retains_two_and_restores_adam_rng_next_update(self):
        import mlx.core as mx
        import mlx.optimizers as optim
        import calibration_v2_runtime as m
        mx.set_default_device(mx.cpu)
        params={'weight':mx.array([1.,2.],dtype=mx.float32)}
        opt=optim.Adam(learning_rate=1e-7,betas=[.9,.999],eps=1e-8,bias_correction=True)
        mx.random.seed(123)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for step in range(1,4):
                params=opt.apply_gradients({'weight':mx.array([.5,-.25])},params);mx.eval(params,opt.state)
                m.save_restart(root,params,opt,{'arm':'U','cursor':step,'bindings':{'test':'fixed'}})
            self.assertEqual([p.name for p in sorted(root.glob('step-*'))],['step-002','step-003'])
            expected=opt.apply_gradients({'weight':mx.array([.3,.2])},params);mx.eval(expected,opt.state)
            expected_rng=mx.random.uniform(shape=(3,));mx.eval(expected_rng)
            restored,state=m.load_restart(root/'step-003',{'test':'fixed'},'U')
            other=optim.Adam(learning_rate=1e-7,betas=[.9,.999],eps=1e-8,bias_correction=True)
            other.state=state
            actual=other.apply_gradients({'weight':mx.array([.3,.2])},restored);mx.eval(actual,other.state)
            self.assertEqual(actual['weight'].tolist(),expected['weight'].tolist())
            self.assertEqual(other.step.item(),4)
            self.assertEqual(mx.random.uniform(shape=(3,)).tolist(),expected_rng.tolist())
            with self.assertRaises(ValueError):m.load_restart(root/'step-003',{'test':'different'},'U')
            with (root/'step-003'/'state.safetensors').open('ab') as out:out.write(b'corrupt')
            with self.assertRaises(ValueError):m.load_restart(root/'step-003',{'test':'fixed'},'U')
