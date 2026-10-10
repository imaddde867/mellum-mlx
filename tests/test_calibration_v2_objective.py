import importlib.util
import math
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


class ObjectiveTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('calibration_v2_objective'), 'objective runner missing')
        import calibration_v2_objective
        return calibration_v2_objective

    def test_next_token_regions_are_not_shifted_again_and_padding_is_excluded(self):
        m=self.module()
        row={'tokens':[10,20,30,40], 'regions':['context','reasoning','final']}
        self.assertEqual(m.target_mask(row,'U',5),[True,True,True,False,False])
        self.assertEqual(m.target_mask(row,'S',5),[False,True,True,False,False])
        for bad in [{'tokens':[10,20,30], 'regions':['final']},
                    {'tokens':[10,20], 'regions':['unknown']}]:
            with self.assertRaises(ValueError): m.target_mask(bad,'S')

    def test_multiturn_boundaries_keep_user_and_tool_results_as_context(self):
        m=self.module()
        from calibration_v2 import regions
        text=('<|im_start|>user\nrequest<|im_end|>\n'
              '<|im_start|>assistant\n<think>\ninspect\n</think>\n\n'
              '<tool_call>\n{"name":"read_file","arguments":{"path":"task.py"}}\n</tool_call><|im_end|>\n'
              '<|im_start|>user\n<tool_response>\nfile result\n</tool_response><|im_end|>\n'
              '<|im_start|>assistant\n<think>\nrepair\n</think>\n\nanswer<|im_end|>\n')
        offsets=[(i,i+1) for i in range(len(text))]
        row={'tokens':list(range(len(text))),'regions':regions(text,offsets,transition_radius=0)}
        mask=m.target_mask(row,'S')
        for literal,want in [('request',False),('file result',False),('inspect',True),
                             ('repair',True),('answer',True),('"read_file"',True)]:
            self.assertEqual(mask[text.index(literal)-1],want)
        self.assertEqual(len(mask),len(text)-1)

    def test_zero_valid_targets_are_rejected(self):
        m=self.module()
        with self.assertRaises(ValueError):m.target_mask({'tokens':[1,2],'regions':['context']},'S')
        with self.assertRaises(ValueError):m.target_mask({'tokens':[1],'regions':[]},'U')

    def measurement(self,assistant,region=None,context=100.):
        return [{'assistant':v,'reasoning':v if region is None else region,
                 'transition':v,'final':v,'tool_call':v,'context':context,'aggregate':context}
                for v in assistant]

    def test_gate_requires_primary_gain_beyond_variation_and_no_region_regression(self):
        m=self.module()
        base=self.measurement([.10,.102,.101,.10,.101])
        self.assertTrue(m.fidelity_gate(base,self.measurement([.08,.081,.08,.08,.081])))
        self.assertFalse(m.fidelity_gate(base,self.measurement([.099,.101,.099,.10,.10])))
        self.assertFalse(m.fidelity_gate(base,self.measurement([.08]*5,region=.13)))
        self.assertTrue(m.fidelity_gate(base,self.measurement([.08]*5,context=200.)))

    def test_gate_rejects_nonfinite_or_incomplete_repeats(self):
        m=self.module()
        base=self.measurement([.1]*5)
        for bad in [self.measurement([.08]*4),self.measurement([.08,.08,math.nan,.08,.08])]:
            with self.assertRaises(ValueError):m.fidelity_gate(base,bad)


    def arm_receipt(self, baseline, checkpoints):
        return {'status':'complete','updates':128,'evaluations':{
            str(step):[{'means':v} for v in self.measurement(values)]
            for step,values in {0:baseline,**checkpoints}.items()}}

    def test_selection_requires_both_pristine_references_and_clear_S_over_U(self):
        m=self.module()
        self.assertTrue(callable(getattr(m,'choose_candidate',None)),'locked selection missing')
        arms={'U':self.arm_receipt([.1]*5,{8:[.08]*5}),
              'S':self.arm_receipt([.1]*5,{8:[.079,.081,.079,.080,.079]})}
        self.assertEqual(m.choose_candidate(arms)['arm'],'U')
        arms['S']=self.arm_receipt([.1]*5,{8:[.06]*5})
        self.assertEqual(m.choose_candidate(arms)['arm'],'S')
        arms={'U':self.arm_receipt([.1]*5,{8:[.09]*5}),
              'S':self.arm_receipt([.08]*5,{8:[.09]*5})}
        self.assertIsNone(m.choose_candidate(arms)['arm'])
        with self.assertRaises(ValueError):
            m.choose_candidate({'U':{**arms['U'],'updates':8},'S':arms['S']})

    def test_incomplete_arm_decision_never_locks_a_candidate(self):
        m=self.module()
        arms={'U':self.arm_receipt([.1]*5,{}),'S':self.arm_receipt([.1]*5,{})}
        arms['U'].update(status='failed',updates=7,error='CUDA graph cache')
        decision=m.experiment_decision(arms)
        self.assertEqual(decision['status'],'blocked')
        self.assertIsNone(decision['arm'])
        self.assertFalse(decision['reserved_opened'])
        self.assertEqual(decision['completed_updates'],{'U':7,'S':128})


@unittest.skipUnless(sys.platform=='linux' and importlib.util.find_spec('mlx'),'numerical tests run on CUDA host')
class NumericalObjectiveTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('calibration_v2_objective'),'objective runner missing')
        import calibration_v2_objective
        return calibration_v2_objective

    def test_masked_values_do_not_change_scalar_or_gradient_and_normalize_by_included_count(self):
        import mlx.core as mx
        mx.set_default_device(mx.cpu)
        m=self.module()
        mask=mx.array([[False,True,True,False]])
        losses=mx.array([[999.,2.,4.,float('nan')]])
        self.assertEqual(m.masked_mean(losses,mask).item(),3.)
        changed=mx.array([[-999999.,2.,4.,900000.]])
        self.assertEqual(m.masked_mean(changed,mask).item(),3.)
        gradients=mx.grad(lambda x:m.masked_mean(x,mask))(changed)
        self.assertEqual(gradients.tolist(),[[0.,.5,.5,0.]])
        with self.assertRaises(ValueError):m.masked_mean(losses,mx.zeros_like(mask))

    def test_fp32_kl_matches_verified_scorer_and_keeps_context_in_input(self):
        import mlx.core as mx
        mx.set_default_device(mx.cpu)
        m=self.module()
        from calibration_v2 import reduce_captured
        teacher=mx.array([[[2*math.log(.75),2*math.log(.25)]]*2],dtype=mx.bfloat16)
        student=mx.zeros_like(teacher)
        labels=['context','final']
        expected=reduce_captured(student,teacher,labels)['fp32']['final'][0]
        losses=m.fp32_kl(student,teacher)
        self.assertAlmostEqual(m.masked_mean(losses,mx.array([[False,True]])).item(),expected,places=6)
        # A scored output can depend on an earlier, unscored context contribution.
        grad=mx.grad(lambda x:m.masked_mean(mx.array([[0.,1.]])+x.sum(),mx.array([[False,True]])))(mx.array([2.,3.]))
        self.assertEqual(grad.tolist(),[1.,1.])

    def test_checkpoint_wrapper_preserves_loss_and_gradients(self):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm.tuner.trainer import grad_checkpoint
        mx.set_default_device(mx.cpu)
        class Block(nn.Module):
            def __init__(self):super().__init__(); self.scale=mx.array([2.])
            def __call__(self,x):return x*self.scale
        model=Block()
        fn=nn.value_and_grad(model,lambda model,x:mx.square(model(x)).sum())
        before,grad_before=fn(model,mx.array([3.]))
        grad_checkpoint(model)
        after,grad_after=fn(model,mx.array([3.]))
        self.assertEqual(before.item(),36.)
        self.assertEqual(after.item(),36.)
        self.assertEqual(grad_before['scale'].tolist(),[36.])
        self.assertEqual(grad_after['scale'].tolist(),[36.])


if __name__=='__main__':unittest.main()
