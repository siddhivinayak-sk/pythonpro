"""Configuration for the conversational model, its training loop and decoding."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict

TORCH = "torch"
TENSORFLOW = "tensorflow"
BACKENDS = (TORCH, TENSORFLOW)

RNN = "rnn"
TRANSFORMER = "transformer"
ARCHITECTURES = (RNN, TRANSFORMER)


@dataclass
class ModelConfig:
    """Shape of the encoder/decoder network. Defaults are sized for a laptop CPU."""

    embedding_dim: int = 128
    hidden_dim: int = 192
    num_layers: int = 1
    dropout: float = 0.1
    use_attention: bool = True
    max_source_length: int = 16
    max_target_length: int = 20
    # "rnn" = GRU encoder/decoder, "transformer" = self-attention encoder/decoder
    architecture: str = RNN
    num_heads: int = 4
    feedforward_dim: int = 512
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2

    def __post_init__(self) -> None:
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {ARCHITECTURES}, got '{self.architecture}'.")
        if self.architecture == TRANSFORMER and self.embedding_dim % self.num_heads:
            raise ValueError(
                f"embedding_dim ({self.embedding_dim}) must be divisible by num_heads ({self.num_heads})."
            )


@dataclass
class TrainingConfig:
    epochs: int = 80
    batch_size: int = 32
    learning_rate: float = 2e-3
    teacher_forcing_ratio: float = 0.9
    gradient_clip: float = 1.0
    validation_fraction: float = 0.1
    patience: int = 15
    seed: int = 42
    log_every: int = 5


@dataclass
class GenerationConfig:
    max_length: int = 24
    strategy: str = "greedy"  # "greedy" | "sampling"
    temperature: float = 0.8
    top_k: int = 5


@dataclass
class ChatbotConfig:
    backend: str = TORCH
    corpus_path: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got '{self.backend}'.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ChatbotConfig":
        return cls(
            backend=payload.get("backend", TORCH),
            corpus_path=payload.get("corpus_path", ""),
            model=ModelConfig(**payload.get("model", {})),
            training=TrainingConfig(**payload.get("training", {})),
            generation=GenerationConfig(**payload.get("generation", {})),
        )
