import gc
import os
import tempfile
import unittest
import uuid
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from safetensors.torch import load_file, save_file

from shm_manager.torch import share_cpu_state_dict


class DirectSharedWeightsTests(unittest.TestCase):
    def test_qwen_native_reader_is_default_and_publishes_loaded_values(self):
        from vllm.config import LoadConfig
        from vllm.model_executor.model_loader import shared_memory_loader as loader_module
        from shm_manager.safetensors_reader import load_qwen_regions

        model = torch.nn.Module()
        model.model = torch.nn.Module()
        model.model.embed_tokens = torch.nn.Embedding(8, 32, dtype=torch.bfloat16)
        model.model.norm = torch.nn.Module()
        model.model.norm.weight = torch.nn.Parameter(torch.empty(32, dtype=torch.bfloat16))
        originals = {
            name: torch.randn(parameter.shape, generator=torch.Generator().manual_seed(i)).to(torch.bfloat16)
            for i, (name, parameter) in enumerate(model.named_parameters())
        }
        with tempfile.TemporaryDirectory(prefix="qwen-default-reader-") as folder:
            save_file(originals, os.path.join(folder, "model.safetensors"))
            model_config = SimpleNamespace(
                model=folder, dtype=torch.bfloat16, quantization=None,
                get_total_num_kv_heads=lambda: 2,
                hf_config=SimpleNamespace(model_type="qwen3", tie_word_embeddings=False,
                                          num_key_value_heads=2))
            load_config = LoadConfig(load_format="shm", use_tqdm_on_load=False)
            config = SimpleNamespace(
                model_config=model_config, load_config=load_config,
                device_config=SimpleNamespace(device="cpu"),
                parallel_config=SimpleNamespace(pipeline_parallel_size=1))
            manifest = os.path.join(folder, "shm_weights.json")

            def postprocess(loaded, *_):
                self.assertFalse(os.path.exists(manifest))
                for name, parameter in loaded.named_parameters():
                    self.assertTrue(torch.equal(parameter, originals[name]))

            # Even an inherited old opt-out must not select the slower reader.
            with patch.dict(os.environ, {"VLLM_CPU_NATIVE_SAFETENSORS": "0"}), patch.object(
                loader_module, "get_tensor_model_parallel_rank", return_value=0,
            ), patch.object(loader_module, "get_tensor_model_parallel_world_size", return_value=1), patch.object(
                loader_module, "get_tp_group", return_value=SimpleNamespace(barrier=Mock()),
            ), patch.object(loader_module, "initialize_model", return_value=model), patch.object(
                loader_module, "process_weights_after_loading", side_effect=postprocess,
            ), patch.object(loader_module.envs, "VLLM_CPU_USE_AMX_GEMM_BF16", True), patch.object(
                loader_module.DefaultModelLoader, "load_weights", side_effect=AssertionError("unexpected generic reader"),
            ), patch("shm_manager.safetensors_reader.load_qwen_regions", wraps=load_qwen_regions) as reader:
                loaded = loader_module.SharedMemoryLoader(load_config).load_model(vllm_config=config)
            try:
                reader.assert_called_once()
                self.assertTrue(os.path.isfile(manifest))
                for name, parameter in loaded.named_parameters():
                    self.assertTrue(torch.equal(parameter, originals[name]))
            finally:
                os.unlink(loaded._shared_weight_path)

    def test_native_safetensors_loader_publishes_after_postprocessing(self):
        from vllm.config import LoadConfig
        from vllm.model_executor.model_loader import shared_memory_loader as loader_module
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod

        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.empty(32, 32, dtype=torch.bfloat16),
                                                 requires_grad=False)
                self.quant_method = UnquantizedLinearMethod()

            def load_weights(self, weights):
                loaded = set()
                for name, tensor in weights:
                    self.get_parameter(name).data.copy_(tensor)
                    loaded.add(name)
                return loaded

        with tempfile.TemporaryDirectory(prefix="direct-loader-") as folder:
            original = torch.randn(32, 32, generator=torch.Generator().manual_seed(7)).to(torch.bfloat16)
            save_file({"weight": original}, os.path.join(folder, "model.safetensors"))
            model_config = SimpleNamespace(model=folder, dtype=torch.bfloat16,
                                           revision=None, quantization=None,
                                           get_total_num_kv_heads=lambda: 2)
            load_config = LoadConfig(load_format="shm", use_tqdm_on_load=False)
            config = SimpleNamespace(model_config=model_config, load_config=load_config,
                                     device_config=SimpleNamespace(device="cpu"),
                                     parallel_config=SimpleNamespace(pipeline_parallel_size=1))
            model = TestModel()
            manifest_path = os.path.join(folder, "shm_weights.json")

            real_postprocess = loader_module.process_weights_after_loading

            def postprocess(model, *args):
                self.assertFalse(os.path.exists(manifest_path))
                self.assertTrue(torch.equal(model.weight, original))
                real_postprocess(model, *args)

            with patch.object(loader_module, "get_tensor_model_parallel_rank", return_value=0), patch.object(
                loader_module, "get_tensor_model_parallel_world_size", return_value=1,
            ), patch.object(loader_module, "get_tp_group", return_value=SimpleNamespace(barrier=Mock())), patch.object(
                loader_module, "initialize_model", return_value=model,
            ), patch.object(loader_module, "process_weights_after_loading", side_effect=postprocess), patch.object(
                loader_module.envs, "VLLM_CPU_USE_AMX_GEMM_BF16", True,
            ):
                loaded = loader_module.SharedMemoryLoader(load_config).load_model(vllm_config=config)
            try:
                activation = torch.randn(32, 32, generator=torch.Generator().manual_seed(11)).to(torch.bfloat16)
                output = loaded.cpu_linear(activation, loaded.weight, None)
                reference = torch.nn.functional.linear(activation, original)
                torch.testing.assert_close(output, reference, rtol=0.03, atol=0.125)
                with open(manifest_path) as stream:
                    manifest = json.load(stream)
                self.assertEqual(manifest["cpu_tp_size"], 1)
                self.assertEqual(manifest["tensor_index"]["weight"][2], [32, 32])
                self.assertFalse(os.path.exists(os.path.join(folder, "rank_0", "tensor.data_0")))
            finally:
                os.unlink(loaded._shared_weight_path)

    def test_safetensors_to_shared_memory_without_serialization(self):
        model_path = "/tmp/direct-weights-" + uuid.uuid4().hex + "/rank_0"
        original = {
            "weight": torch.arange(96).reshape(8, 12).to(torch.bfloat16),
            "norm": torch.ones(12, dtype=torch.float32),
        }
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = os.path.join(folder, "model.safetensors")
            save_file(original, checkpoint)
            loaded = load_file(checkpoint, device="cpu")
            shared, metadata, shm_path = share_cpu_state_dict(loaded, model_path)
            try:
                for name in original:
                    self.assertTrue(torch.equal(shared[name], original[name]))
                    self.assertEqual(metadata[name][2], tuple(original[name].shape))
                self.assertEqual(os.listdir(folder), ["model.safetensors"])
                with self.assertRaises(FileExistsError):
                    share_cpu_state_dict(loaded, model_path)
                del shared["norm"]
                gc.collect()
                self.assertTrue(torch.equal(shared["weight"], original["weight"]))
                shared["weight"].zero_()
                self.assertTrue(torch.equal(loaded["weight"], original["weight"]))
            finally:
                os.unlink(shm_path)

    def test_noncontiguous_view_copies_only_logical_values(self):
        tensor = torch.arange(48, dtype=torch.float32).reshape(6, 8)[:, ::2]
        shared, metadata, shm_path = share_cpu_state_dict(
            {"view": tensor}, "/tmp/direct-view-" + uuid.uuid4().hex + "/rank_0")
        try:
            self.assertTrue(torch.equal(shared["view"], tensor))
            self.assertTrue(shared["view"].is_contiguous())
            self.assertEqual(metadata["view"][1], tensor.numel() * tensor.element_size())
        finally:
            os.unlink(shm_path)

    def test_empty_state_rejected_before_allocation(self):
        with self.assertRaises(ValueError):
            share_cpu_state_dict({}, "/tmp/unused/rank_0")
        with self.assertRaises(ValueError):
            share_cpu_state_dict({"empty": torch.empty(0)}, "/tmp/unused/rank_0")

    def test_native_dense_layout_has_no_inter_tensor_padding(self):
        tensors = {"first": torch.ones(3, dtype=torch.bfloat16),
                   "second": torch.ones(5, dtype=torch.bfloat16)}
        shared, metadata, path = share_cpu_state_dict(
            tensors, "/tmp/direct-dense-" + uuid.uuid4().hex + "/rank_0", alignment=1)
        try:
            self.assertEqual(metadata["second"][0], 6)
            self.assertEqual(os.stat(path).st_size, 16)
            self.assertTrue(torch.equal(shared["second"], tensors["second"]))
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
