"""Run in a PyTorch environment; tiny CPU tensors, no model download required."""
import copy
import unittest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from chunked_mlp import patch_mlp, native_checkpoint


class LoRALinear(nn.Module):
    def __init__(self, width_in, width_out):
        super().__init__()
        self.base = nn.Linear(width_in, width_out, bias=False)
        self.base.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.randn(3, width_in) * .1)
        self.lora_B = nn.Parameter(torch.randn(width_out, 3) * .1)

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.T) @ self.lora_B.T


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = LoRALinear(8, 19)
        self.up_proj = LoRALinear(8, 19)
        self.down_proj = LoRALinear(19, 8)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class ChunkTests(unittest.TestCase):
    def test_unsloth_checkpoint_override_bypassed(self):
        from unittest.mock import patch
        import torch.utils.checkpoint as ck
        original = ck.checkpoint
        def shim(*args, **kwargs):
            raise AssertionError('Unsloth shim must not be used for inner checkpoint')
        with patch.object(ck, '_unsloth_pristine_checkpoint', original, create=True):
            with patch.object(ck, 'checkpoint', shim):
                self.assertIs(native_checkpoint(), original)
                self.compare()

    def compare(self, outer_checkpoint=False, input_grad=True, hooked=False):
        torch.manual_seed(37)
        reference = MLP().double()
        actual = copy.deepcopy(reference)
        if hooked:
            actual._hf_hook = object()
            actual._old_forward = actual.forward
            def dispatch(x):
                actual.hook_calls += 1
                return actual._old_forward(x)
            actual.hook_calls = 0
            actual.forward = dispatch
        names = set(actual.state_dict())
        patch_mlp(actual, 4)
        self.assertEqual(names, set(actual.state_dict()))
        x = torch.randn(2, 11, 8, dtype=torch.float64, requires_grad=input_grad)
        y = x.detach().clone().requires_grad_(input_grad)
        a = reference(x)
        b = checkpoint(actual, y, use_reentrant=True) if outer_checkpoint else actual(y)
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)
        a.square().sum().backward()
        b.square().sum().backward()
        if input_grad:
            torch.testing.assert_close(x.grad, y.grad, atol=1e-10, rtol=1e-9)
        for (name, p), (_, q) in zip(reference.named_parameters(), actual.named_parameters()):
            if p.requires_grad:
                self.assertIsNotNone(q.grad, name)
                torch.testing.assert_close(p.grad, q.grad, atol=1e-10, rtol=1e-9)
        if hooked:
            self.assertGreater(actual.hook_calls, 0)

    def test_output_input_and_lora_gradients(self): self.compare()
    def test_nested_outer_checkpoint(self): self.compare(outer_checkpoint=True)
    def test_frozen_input_still_trains_lora(self): self.compare(input_grad=False)
    def test_dispatch_wrapper_preserved(self): self.compare(hooked=True, outer_checkpoint=True)

    def test_saved_wide_activations_not_accumulated(self):
        model = MLP()
        x = torch.randn(1, 11, 8, requires_grad=True)
        def saved_shapes():
            shapes = []
            def pack(t):
                shapes.append(tuple(t.shape))
                return t
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
                model(x)
            return shapes
        self.assertTrue(any(s[-1] == 19 for s in saved_shapes()))
        patch_mlp(model, 4)
        self.assertFalse(any(s and s[-1] == 19 for s in saved_shapes()))


if __name__ == '__main__': unittest.main()
