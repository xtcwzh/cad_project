import unittest
from types import SimpleNamespace
from device_plan import build_device_plan, validate_device_plan, validate_execution_hooks


class DevicePlanTests(unittest.TestCase):
    def setUp(self):
        self.config = {'model_type': 'qwen3_5', 'text_config': {'num_hidden_layers': 64}}
        self.mapping = build_device_plan(self.config, [12, 20, 20, 12], 4)

    def test_boundaries_and_full_coverage(self):
        layers = [self.mapping[f'model.language_model.layers.{i}'] for i in range(64)]
        self.assertEqual(layers, [0] * 12 + [1] * 20 + [2] * 20 + [3] * 12)
        self.assertEqual(self.mapping['lm_head'], 3)
        self.assertEqual(self.mapping['model.visual'], 0)
        self.assertNotIn('', self.mapping)
        self.assertEqual(self.mapping['model.language_model.rotary_emb'], 0)

    def test_stale_nested_execution_hook_rejected(self):
        name = 'model.language_model.layers.12.input_layernorm'
        hook = SimpleNamespace(execution_device='cuda:0')
        module = SimpleNamespace(_hf_hook=SimpleNamespace(hooks=[hook]))
        model = SimpleNamespace(named_modules=lambda: [(name, module)])
        with self.assertRaisesRegex(RuntimeError, 'Execution hook mismatch'):
            validate_execution_hooks(model, self.mapping)
        hook.execution_device = 'cuda:1'
        validate_execution_hooks(model, self.mapping)

    def test_head_only_last_gpu(self):
        mapping = build_device_plan(self.config, [16, 24, 24, 0], 4)
        layers = [mapping[f'model.language_model.layers.{i}'] for i in range(64)]
        self.assertEqual(layers, [0] * 16 + [1] * 24 + [2] * 24)
        self.assertEqual(mapping['lm_head'], 3)
        self.assertEqual(mapping['model.language_model.norm'], 3)
        self.assertEqual(set(mapping.values()), {0, 1, 2, 3})
        with self.assertRaises(ValueError):
            build_device_plan(self.config, [16, 24, 25, -1], 4)

    def test_invalid_layout_rejected(self):
        for counts in ([12, 20, 20, 11], [32, 32], [0, 20, 20, 24], [12., 20, 20, 12]):
            with self.assertRaises(ValueError): build_device_plan(self.config, counts, 4)
        self.config['text_config']['tie_word_embeddings'] = True
        with self.assertRaises(ValueError): build_device_plan(self.config, [12, 20, 20, 12], 4)

    def test_actual_placement_and_adapter_wrapper(self):
        mapping = self.mapping
        devices = {name: SimpleNamespace(type='cuda', index=gpu) for name, gpu in mapping.items()}
        class Model:
            def named_modules(self): return [(name, object()) for name in mapping]
            def named_parameters(self):
                return [(name + '.lora_A.default.weight', SimpleNamespace(device=device))
                        for name, device in devices.items() if name]
        model = Model()
        validate_device_plan(SimpleNamespace(get_base_model=lambda: model), mapping)
        devices['model.language_model.layers.52'].index = 2
        with self.assertRaisesRegex(RuntimeError, 'Placement mismatch'):
            validate_device_plan(model, mapping)

    def test_wrong_architecture_paths_rejected(self):
        model = SimpleNamespace(named_modules=lambda: [('', object())])
        with self.assertRaisesRegex(RuntimeError, 'module paths'):
            validate_device_plan(model, self.mapping)


if __name__ == '__main__': unittest.main()
