"""Backward-compatible exports for model configuration types.

The configuration implementation now lives in :mod:`vllm.config`, but a
number of downstream integrations still import these names from
``vllm.config.model``.
"""

from . import ModelConfig, ModelDType, TokenizerMode

__all__ = ["ModelConfig", "ModelDType", "TokenizerMode"]
