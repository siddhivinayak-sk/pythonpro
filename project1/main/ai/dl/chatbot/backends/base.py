"""Common interface so the training loop is identical for PyTorch and TensorFlow."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..config import ModelConfig
from ..dataset import ConversationDataset


@dataclass
class EpochLog:
    epoch: int
    train_loss: float
    validation_loss: Optional[float]
    perplexity: float
    seconds: float

    def format(self) -> str:
        validation = "   -   " if self.validation_loss is None else f"{self.validation_loss:7.4f}"
        return (
            f"epoch {self.epoch:>4}  train {self.train_loss:7.4f}  "
            f"val {validation}  ppl {self.perplexity:8.2f}  {self.seconds:5.2f}s"
        )


class Seq2SeqBackend(ABC):
    """Encoder/decoder model plus the per-epoch operations the trainer drives."""

    name: str = "base"

    def __init__(self, vocab_size: int, config: ModelConfig, seed: int = 42) -> None:
        self.vocab_size = vocab_size
        self.config = config
        self.seed = seed

    @abstractmethod
    def build(self) -> None:
        """Instantiate the network and its optimiser."""

    @abstractmethod
    def set_learning_rate(self, learning_rate: float) -> None:
        ...

    @abstractmethod
    def train_epoch(
        self,
        dataset: ConversationDataset,
        batch_size: int,
        teacher_forcing_ratio: float,
        gradient_clip: float,
        seed: int,
    ) -> float:
        """One pass over the data; returns the mean masked cross-entropy."""

    @abstractmethod
    def evaluate(self, dataset: ConversationDataset, batch_size: int) -> float:
        """Mean masked cross-entropy without updating weights."""

    @abstractmethod
    def generate(
        self,
        source_ids: np.ndarray,
        max_length: int,
        strategy: str = "greedy",
        temperature: float = 0.8,
        top_k: int = 5,
    ) -> List[int]:
        """Decode a reply, one token at a time, until <eos>."""

    @abstractmethod
    def parameter_count(self) -> int:
        """Number of trainable parameters."""

    @abstractmethod
    def save_weights(self, directory: Path) -> None:
        ...

    @abstractmethod
    def load_weights(self, directory: Path) -> None:
        ...

    def state_snapshot(self):
        """In-memory copy of the weights, used to keep the best epoch."""
        raise NotImplementedError

    def restore_snapshot(self, snapshot) -> None:
        raise NotImplementedError


def sample_from_logits(logits: np.ndarray, strategy: str, temperature: float, top_k: int) -> int:
    """Shared greedy / top-k sampling so both backends decode identically."""
    if strategy == "greedy":
        return int(np.argmax(logits))

    scaled = logits.astype(np.float64) / max(temperature, 1e-6)
    if top_k > 0:
        keep = np.argpartition(scaled, -min(top_k, scaled.size))[-min(top_k, scaled.size) :]
        masked = np.full_like(scaled, -np.inf)
        masked[keep] = scaled[keep]
        scaled = masked
    scaled -= scaled.max()
    probabilities = np.exp(scaled)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
