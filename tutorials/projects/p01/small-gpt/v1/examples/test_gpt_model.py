import math
import unittest
import torch
from torch.nn import functional as F
from gpt_attention import attend
from gpt_config import Config
from gpt_model import GPT, Block, parameter_formula
from gpt_objective import nll_sum
from gpt_tokenizer import BOS, PAD


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(31)
        self.c = Config(width=16, heads=4, hidden=32, context=8)

    def test_attention_values_and_gradients(self):
        q, k, v = [torch.randn(2,2,3,4, dtype=torch.float64, requires_grad=True) for _ in range(3)]
        mask = torch.ones(3,3,dtype=torch.bool).tril()
        a = attend(q,k,v,mask)
        b = attend(q,k,v,mask,optimized=True)
        torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-10)
        grad_a = torch.autograd.grad(a.square().sum(), (q,k,v), retain_graph=True)
        grad_b = torch.autograd.grad(b.square().sum(), (q,k,v))
        for x,y in zip(grad_a,grad_b):
            torch.testing.assert_close(x,y,rtol=1e-9,atol=1e-9)
        self.assertTrue(torch.autograd.gradcheck(lambda a,b,c: attend(a,b,c,mask), (q,k,v)))

    def test_zero_score_prefix_average(self):
        q = torch.zeros(1,1,3,1)
        v = torch.tensor([2.,4.,8.]).view(1,1,3,1)
        out = attend(q,q,v,torch.ones(3,3,dtype=torch.bool).tril())
        torch.testing.assert_close(out.flatten(),torch.tensor([2.,3.,14/3]))

    def test_full_model_causality_and_reference(self):
        m=GPT(self.c).eval(); ids=torch.tensor([[BOS,1,2,3,4]])
        a=m(ids); changed=ids.clone(); changed[:,3:]=9
        torch.testing.assert_close(a[:,:3],m(changed)[:,:3],rtol=0,atol=0)
        torch.testing.assert_close(a,m(ids,optimized=True),rtol=1e-5,atol=1e-6)

    def test_padding_and_bounds(self):
        m=GPT(self.c).eval()
        prefix=torch.tensor([[BOS,1,2]])
        padded=torch.tensor([[BOS,1,2,PAD,PAD]])
        torch.testing.assert_close(m(prefix),m(padded)[:,:3])
        for bad in [[[PAD]], [[BOS,PAD,1]], [[BOS]*9], [[]]]:
            with self.assertRaises(ValueError): m(torch.tensor(bad,dtype=torch.long))

    def test_zero_prenorm_sublayers_leave_identity(self):
        b=Block(self.c).eval()
        for p in b.parameters(): p.data.zero_()
        x=torch.randn(2,3,self.c.width)
        torch.testing.assert_close(b(x,torch.ones(2,3,dtype=torch.bool)),x,rtol=0,atol=0)

    def test_parameter_count_and_tied_head_gradient(self):
        m=GPT(Config()); self.assertEqual(sum(p.numel() for p in m.parameters()),parameter_formula(Config()))
        self.assertEqual(parameter_formula(Config()),120832)
        loss,n=nll_sum(m(torch.tensor([[BOS,1]])),torch.tensor([[1,2]])); (loss/n).backward()
        self.assertGreater(m.token.weight.grad[250].abs().sum().item(),0) # output use, not looked up
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_nll_and_gradient(self):
        logits=torch.tensor([[[0.,math.log(3)]]],dtype=torch.float64,requires_grad=True)
        loss,n=nll_sum(logits,torch.tensor([[1]])); loss.backward()
        self.assertAlmostEqual(loss.item(),-math.log(.75))
        torch.testing.assert_close(logits.grad,torch.tensor([[[.25,-.25]]],dtype=torch.float64))
        self.assertEqual(n,1)

    def test_unequal_accumulation(self):
        logits=torch.randn(2,3,260,dtype=torch.float64,requires_grad=True)
        targets=torch.tensor([[1,2,3],[4,PAD,PAD]])
        combined,count=nll_sum(logits,targets)
        pieces=[nll_sum(logits[i:i+1],targets[i:i+1])[0] for i in range(2)]
        g1=torch.autograd.grad(combined/count,logits,retain_graph=True)[0]
        g2=torch.autograd.grad(sum(pieces)/4,logits)[0]
        torch.testing.assert_close(g1,g2)
        with self.assertRaises(ValueError): nll_sum(logits,torch.full((2,3),PAD))


if __name__ == "__main__":
    unittest.main()
