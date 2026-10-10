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

    def test_resume_cursor_rejects_wrong_order_budget_or_qualification(self):
        m=self.module();self.assertTrue(callable(getattr(m,'resume_cursor',None)))
        order=['one','two','three']
        receipt={'cursor':2,'update_order':['one','two'],'qualification_sha256':'new-qualified'}
        self.assertEqual(m.resume_cursor(receipt,order,'new-qualified'),2)
        for bad in [{**receipt,'cursor':4},{**receipt,'update_order':['two','one']},
                    {**receipt,'qualification_sha256':'failed-old'}]:
            with self.assertRaises(ValueError):m.resume_cursor(bad,order,'new-qualified')

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
    def test_model_only_setup_does_not_allocate_optimizer_state(self):
        import mlx.core as mx
        import mlx.nn as nn
        import calibration_v2_runtime as m
        mx.set_default_device(mx.cpu)
        class Block(nn.Module):
            def __init__(self):super().__init__();self.scale=mx.array([2.],dtype=mx.bfloat16)
            def __call__(self,x):return x*self.scale
        class Model(nn.Module):
            def __init__(self):super().__init__();self.layers=[Block()]
        self.assertTrue(callable(getattr(m,'model_setup',None)))
        with patch.object(m.base,'student',return_value=(Model(),None,{},{})),patch.object(m,'adam',side_effect=AssertionError('disposable Adam allocated')):
            model,original,_,_=m.model_setup()
            self.assertEqual(model.layers[0].scale.dtype,mx.bfloat16)
            self.assertEqual(original['layers.0.scale'].tolist(),[2.])

    def test_host_reduction_matches_original_fp32_small_arrays(self):
        import mlx.core as mx
        import numpy as np
        import calibration_v2_runtime as m
        mx.set_default_device(mx.cpu)
        self.assertTrue(callable(getattr(m,'host_accumulate',None)))
        values=[np.array([1.25,-2.,3.],dtype=np.float32),np.array([-.5,4.,2.],dtype=np.float32)]*2+[np.array([2.,-1.,.5],dtype=np.float32)]
        total={};gpu=None
        for value in values:
            m.host_accumulate(total,'weight',value)
            gpu=mx.array(value) if gpu is None else gpu+mx.array(value)
        expected=gpu/5;mx.eval(expected)
        mean={'weight':total['weight']/np.float32(5)}
        np.testing.assert_array_equal(mean['weight'],np.array(expected))
        reference={'weight':np.array([1.,2.,3.],dtype=np.float32)}
        gpu_error=(mx.square(expected-mx.array(reference['weight'])).sum()/mx.square(mx.array(reference['weight'])).sum()).item()**.5
        self.assertAlmostEqual(m.host_relative(mean,reference),gpu_error,places=6)

    def test_host_comparison_rejects_nonfinite_and_checks_disk_references(self):
        import numpy as np
        import calibration_v2_runtime as m
        with self.assertRaises(ValueError):m.host_relative({'x':np.array([np.nan],dtype=np.float32)},{'x':np.array([1.],dtype=np.float32)})
        with self.assertRaises(ValueError):m.host_accumulate({},'x',np.array([np.inf],dtype=np.float32))
        with tempfile.TemporaryDirectory() as tmp:
            ref=Path(tmp)/'ref.npy';np.save(ref,np.array([1.,2.],dtype=np.float32))
            self.assertAlmostEqual(m.host_relative({'x':np.array([2.,4.],dtype=np.float32)},{'x':ref}),1.)

    def test_runner_restoration_helper_materializes_bf16_model_and_next_update(self):
        import mlx.core as mx
        import mlx.nn as nn
        import calibration_v2_runtime as m
        mx.set_default_device(mx.cpu)
        class Model(nn.Module):
            def __init__(self):super().__init__();self.weight=mx.array([1.,2.],dtype=mx.bfloat16)
        params={'weight':mx.array([1.,2.],dtype=mx.float32)};opt=m.adam()
        params=opt.apply_gradients({'weight':mx.array([.5,-.25])},params);mx.eval(params,opt.state)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);m.save_restart(root,params,opt,{'arm':'U','cursor':1,'bindings':{'fixed':'source'}})
            model=Model();loaded,restored=m.restore_training_state(model,root/'step-001',{'fixed':'source'},'U')
            self.assertEqual(model.weight.dtype,mx.bfloat16)
            self.assertEqual(model.weight.tolist(),params['weight'].astype(mx.bfloat16).tolist())
            self.assertEqual(loaded['weight'].dtype,mx.float32)
            self.assertEqual(restored.state['weight']['m'].tolist(),opt.state['weight']['m'].tolist())
            with patch.object(restored,'init',side_effect=AssertionError('restored init')):
                updated=restored.apply_gradients({'weight':mx.array([.3,.2])},loaded);mx.eval(updated,restored.state)
            self.assertEqual(restored.step.item(),2)
            expected=opt.apply_gradients({'weight':mx.array([.3,.2])},params);mx.eval(expected)
            self.assertEqual(updated['weight'].tolist(),expected['weight'].tolist())

    def test_restored_optimizer_keeps_moments_and_skips_init(self):
        import mlx.core as mx
        import calibration_v2_runtime as m
        mx.set_default_device(mx.cpu)
        self.assertTrue(callable(getattr(m,'restored_optimizer',None)))
        params={'weight':mx.array([1.,2.],dtype=mx.float32)}
        opt=m.adam();params=opt.apply_gradients({'weight':mx.array([.5,-.25])},params);mx.eval(params,opt.state)
        expected=opt.apply_gradients({'weight':mx.array([.3,.2])},params);mx.eval(expected)
        # Independent step-one fixture; discarding its nonzero moments changes the next update.
        first=m.adam();first_params=first.apply_gradients({'weight':mx.array([.5,-.25])},{'weight':mx.array([1.,2.])});mx.eval(first_params,first.state)
        from mlx.utils import tree_map
        saved_state=tree_map(lambda x:x,first.state)
        restored=m.restored_optimizer(first_params,first.state,1)
        with patch.object(restored,'init',side_effect=AssertionError('restored moments initialized')):
            actual=restored.apply_gradients({'weight':mx.array([.3,.2])},first_params);mx.eval(actual,restored.state)
        self.assertEqual(actual['weight'].tolist(),expected['weight'].tolist())
        self.assertEqual(restored.step.item(),2)
        bad={**saved_state,'weight':{'m':mx.zeros((1,)),'v':mx.zeros((1,))}}
        with self.assertRaises(ValueError):m.restored_optimizer(first_params,bad,1)
        with self.assertRaises(ValueError):m.restored_optimizer(first_params,{**saved_state,'learning_rate':mx.array([1e-7])},1)

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
