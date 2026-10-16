"""PyTorch transformer encoder/decoder, with multi-head attention written out explicitly.

Compared with the GRU backend this model has no recurrence: the whole reply is scored in a
single parallel forward pass during training, and only generation is sequential.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from ..config import ModelConfig
from ..dataset import ConversationDataset, iterate_batches
from ..tokenizer import EOS_ID, PAD_ID, SOS_ID
from .base import Seq2SeqBackend, sample_from_logits

WEIGHTS_FILE = "model.pt"


class PositionalEncoding(nn.Module):
    """Adds the classic sinusoidal position signal, since attention itself is order blind."""

    def __init__(self, model_dim: int, dropout: float, max_length: int = 512) -> None:
        super().__init__()
        self.scale = math.sqrt(model_dim)
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_length).unsqueeze(1).float()
        divisor = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        encoding = torch.zeros(max_length, model_dim)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, embedded: torch.Tensor) -> torch.Tensor:
        return self.dropout(embedded * self.scale + self.encoding[:, : embedded.size(1)])


class MultiHeadAttention(nn.Module):
    """softmax(QK^T / sqrt(d_k)) V, computed for `num_heads` subspaces at once."""

    def __init__(self, model_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.query = nn.Linear(model_dim, model_dim)
        self.key = nn.Linear(model_dim, model_dim)
        self.value = nn.Linear(model_dim, model_dim)
        self.out = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def _split(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,  # (batch, 1, q_len | 1, k_len), True = attend
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, q_len, _ = query.shape
        q, k, v = self._split(self.query(query)), self._split(self.key(key)), self._split(self.value(value))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = self.dropout(torch.softmax(scores, dim=-1))

        context = torch.matmul(weights, v).transpose(1, 2).contiguous().view(batch, q_len, -1)
        return self.out(context), weights


class FeedForward(nn.Module):
    def __init__(self, model_dim: int, feedforward_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(model_dim, feedforward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, model_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    """Self-attention over the prompt, then a position-wise feed forward block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(config.embedding_dim, config.num_heads, config.dropout)
        self.feed_forward = FeedForward(config.embedding_dim, config.feedforward_dim, config.dropout)
        self.norm1 = nn.LayerNorm(config.embedding_dim)
        self.norm2 = nn.LayerNorm(config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
        attended, _ = self.self_attention(x, x, x, source_mask)
        x = self.norm1(x + self.dropout(attended))
        return self.norm2(x + self.dropout(self.feed_forward(x)))


class DecoderLayer(nn.Module):
    """Causal self-attention over the reply, cross-attention to the prompt, then feed forward."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(config.embedding_dim, config.num_heads, config.dropout)
        self.cross_attention = MultiHeadAttention(config.embedding_dim, config.num_heads, config.dropout)
        self.feed_forward = FeedForward(config.embedding_dim, config.feedforward_dim, config.dropout)
        self.norm1 = nn.LayerNorm(config.embedding_dim)
        self.norm2 = nn.LayerNorm(config.embedding_dim)
        self.norm3 = nn.LayerNorm(config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        target_mask: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attended, _ = self.self_attention(x, x, x, target_mask)
        x = self.norm1(x + self.dropout(attended))
        attended, cross_weights = self.cross_attention(x, memory, memory, source_mask)
        x = self.norm2(x + self.dropout(attended))
        return self.norm3(x + self.dropout(self.feed_forward(x))), cross_weights


def padding_mask(sequence: torch.Tensor) -> torch.Tensor:
    """(batch, 1, 1, length) - True where the token is real."""
    mask = (sequence != PAD_ID).unsqueeze(1).unsqueeze(2)
    # A prompt of only padding would make the softmax NaN.
    mask[:, :, :, 0] = True
    return mask


def causal_mask(length: int, device: torch.device) -> torch.Tensor:
    """(1, 1, length, length) - position t may only attend to positions <= t."""
    return torch.tril(torch.ones(length, length, dtype=torch.bool, device=device)).unsqueeze(0).unsqueeze(0)


class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size: int, config: ModelConfig) -> None:
        super().__init__()
        # Prompts and replies are the same language, so the embedding table is shared.
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim, padding_idx=PAD_ID)
        self.positional = PositionalEncoding(config.embedding_dim, config.dropout)
        self.encoder_layers = nn.ModuleList(EncoderLayer(config) for _ in range(config.num_encoder_layers))
        self.decoder_layers = nn.ModuleList(DecoderLayer(config) for _ in range(config.num_decoder_layers))
        self.output = nn.Linear(config.embedding_dim, vocab_size)

    def encode(self, source: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
        x = self.positional(self.embedding(source))
        for layer in self.encoder_layers:
            x = layer(x, source_mask)
        return x

    def decode(
        self,
        target_inputs: torch.Tensor,
        memory: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.positional(self.embedding(target_inputs))
        mask = causal_mask(target_inputs.size(1), target_inputs.device) & (
            target_inputs != PAD_ID
        ).unsqueeze(1).unsqueeze(2)
        mask[:, :, :, 0] = True
        for layer in self.decoder_layers:
            x, _ = layer(x, memory, mask, source_mask)
        return self.output(x)

    def forward(self, source: torch.Tensor, target_inputs: torch.Tensor) -> torch.Tensor:
        source_mask = padding_mask(source)
        return self.decode(target_inputs, self.encode(source, source_mask), source_mask)


class TorchTransformer(Seq2SeqBackend):
    name = "torch-transformer"

    def build(self) -> None:
        torch.manual_seed(self.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TransformerSeq2Seq(self.vocab_size, self.config).to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)

    def set_learning_rate(self, learning_rate: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

    def _tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array).to(self.device)

    def train_epoch(
        self,
        dataset: ConversationDataset,
        batch_size: int,
        teacher_forcing_ratio: float,
        gradient_clip: float,
        seed: int,
    ) -> float:
        # teacher_forcing_ratio is unused: a transformer sees the whole shifted reply at once.
        self.model.train()
        total_loss, batches = 0.0, 0
        for source, decoder_input, target in iterate_batches(dataset, batch_size, shuffle=True, seed=seed):
            self.optimizer.zero_grad()
            logits = self.model(self._tensor(source), self._tensor(decoder_input))
            loss = self.criterion(logits.reshape(-1, self.vocab_size), self._tensor(target).reshape(-1))
            loss.backward()
            if gradient_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
            self.optimizer.step()
            total_loss += float(loss.item())
            batches += 1
        return total_loss / max(batches, 1)

    @torch.no_grad()
    def evaluate(self, dataset: ConversationDataset, batch_size: int) -> float:
        if len(dataset) == 0:
            return float("nan")
        self.model.eval()
        total_loss, batches = 0.0, 0
        for source, decoder_input, target in iterate_batches(dataset, batch_size, shuffle=False):
            logits = self.model(self._tensor(source), self._tensor(decoder_input))
            loss = self.criterion(logits.reshape(-1, self.vocab_size), self._tensor(target).reshape(-1))
            total_loss += float(loss.item())
            batches += 1
        return total_loss / max(batches, 1)

    @torch.no_grad()
    def generate(
        self,
        source_ids: np.ndarray,
        max_length: int,
        strategy: str = "greedy",
        temperature: float = 0.8,
        top_k: int = 5,
    ) -> List[int]:
        self.model.eval()
        source = self._tensor(np.asarray(source_ids, dtype=np.int64).reshape(1, -1))
        source_mask = padding_mask(source)
        memory = self.model.encode(source, source_mask)

        tokens = [SOS_ID]
        generated: List[int] = []
        for _ in range(max_length):
            decoder_input = torch.tensor([tokens], dtype=torch.long, device=self.device)
            logits = self.model.decode(decoder_input, memory, source_mask)
            next_id = sample_from_logits(logits[0, -1].cpu().numpy(), strategy, temperature, top_k)
            if next_id == EOS_ID:
                break
            generated.append(next_id)
            tokens.append(next_id)
        return generated

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)

    def save_weights(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), directory / WEIGHTS_FILE)

    def load_weights(self, directory: Path) -> None:
        self.model.load_state_dict(torch.load(directory / WEIGHTS_FILE, map_location=self.device))
        self.model.eval()

    def state_snapshot(self) -> dict:
        return {key: value.detach().clone() for key, value in self.model.state_dict().items()}

    def restore_snapshot(self, snapshot: dict) -> None:
        self.model.load_state_dict(snapshot)
