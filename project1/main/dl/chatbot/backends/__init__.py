"""Backend implementations. Each one is imported lazily so the other stays optional."""

from __future__ import annotations

import importlib.util
from typing import List

from ..config import TENSORFLOW, TORCH, TRANSFORMER
from .base import EpochLog, Seq2SeqBackend


def is_available(backend: str) -> bool:
    module = "torch" if backend == TORCH else "tensorflow"
    return importlib.util.find_spec(module) is not None


def available_backends() -> List[str]:
    return [backend for backend in (TORCH, TENSORFLOW) if is_available(backend)]


def create_backend(backend: str, vocab_size: int, config, seed: int = 42) -> Seq2SeqBackend:
    """Pick the implementation for the requested framework and architecture."""
    transformer = getattr(config, "architecture", "rnn") == TRANSFORMER
    if backend == TORCH:
        if transformer:
            from .torch_transformer import TorchTransformer

            return TorchTransformer(vocab_size, config, seed)
        from .torch_backend import TorchSeq2Seq

        return TorchSeq2Seq(vocab_size, config, seed)
    if backend == TENSORFLOW:
        if transformer:
            from .tf_transformer import TensorFlowTransformer

            return TensorFlowTransformer(vocab_size, config, seed)
        from .tf_backend import TensorFlowSeq2Seq

        return TensorFlowSeq2Seq(vocab_size, config, seed)
    raise ValueError(f"Unknown backend '{backend}'.")


__all__ = ["EpochLog", "Seq2SeqBackend", "available_backends", "create_backend", "is_available"]
