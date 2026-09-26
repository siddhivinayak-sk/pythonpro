"""PyTorch GRU encoder / attention decoder and its training step."""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from ..config import ModelConfig
from ..dataset import ConversationDataset, iterate_batches
from ..tokenizer import EOS_ID, PAD_ID, SOS_ID
from .base import Seq2SeqBackend, sample_from_logits

WEIGHTS_FILE = "model.pt"


class Encoder(nn.Module):
    """Embeds the prompt and runs a GRU over it, keeping every step for attention."""

    def __init__(self, vocab_size: int, config: ModelConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim, padding_idx=PAD_ID)
        self.dropout = nn.Dropout(config.dropout)
        self.gru = nn.GRU(
            config.embedding_dim,
            config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )

    def forward(self, source: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(source))
        outputs, hidden = self.gru(embedded)
        return outputs, hidden


class LuongAttention(nn.Module):
    """score(h_dec, h_enc) = h_dec^T W h_enc, masked over padding."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.project = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,  # (batch, hidden)
        encoder_outputs: torch.Tensor,  # (batch, source_len, hidden)
        mask: torch.Tensor,  # (batch, source_len) True where real tokens
    ) -> torch.Tensor:
        scores = torch.bmm(encoder_outputs, self.project(decoder_hidden).unsqueeze(2)).squeeze(2)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(1)
        return torch.bmm(weights, encoder_outputs).squeeze(1)


class Decoder(nn.Module):
    """Generates the reply one token at a time, optionally attending to the encoder."""

    def __init__(self, vocab_size: int, config: ModelConfig) -> None:
        super().__init__()
        self.use_attention = config.use_attention
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim, padding_idx=PAD_ID)
        self.dropout = nn.Dropout(config.dropout)
        self.gru = nn.GRU(
            config.embedding_dim,
            config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.attention = LuongAttention(config.hidden_dim) if config.use_attention else None
        output_dim = config.hidden_dim * 2 if config.use_attention else config.hidden_dim
        self.output = nn.Linear(output_dim, vocab_size)

    def step(
        self,
        token: torch.Tensor,  # (batch,)
        hidden: torch.Tensor,  # (layers, batch, hidden)
        encoder_outputs: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(token.unsqueeze(1)))
        output, hidden = self.gru(embedded, hidden)
        state = output.squeeze(1)
        if self.attention is not None:
            context = self.attention(state, encoder_outputs, mask)
            state = torch.cat([state, context], dim=1)
        return self.output(state), hidden


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size: int, config: ModelConfig) -> None:
        super().__init__()
        self.encoder = Encoder(vocab_size, config)
        self.decoder = Decoder(vocab_size, config)

    def forward(
        self,
        source: torch.Tensor,
        decoder_inputs: torch.Tensor,
        teacher_forcing_ratio: float = 1.0,
    ) -> torch.Tensor:
        encoder_outputs, hidden = self.encoder(source)
        mask = source != PAD_ID
        # A prompt of only padding would make the masked softmax NaN.
        mask[:, 0] = True

        steps = decoder_inputs.size(1)
        logits = []
        token = decoder_inputs[:, 0]
        for position in range(steps):
            step_logits, hidden = self.decoder.step(token, hidden, encoder_outputs, mask)
            logits.append(step_logits)
            if position + 1 < steps:
                use_teacher = random.random() < teacher_forcing_ratio
                token = decoder_inputs[:, position + 1] if use_teacher else step_logits.argmax(dim=1)
        return torch.stack(logits, dim=1)


class TorchSeq2Seq(Seq2SeqBackend):
    name = "torch"

    def build(self) -> None:
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Seq2Seq(self.vocab_size, self.config).to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-3)

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
        self.model.train()
        total_loss, batches = 0.0, 0
        for source, decoder_input, target in iterate_batches(dataset, batch_size, shuffle=True, seed=seed):
            self.optimizer.zero_grad()
            logits = self.model(self._tensor(source), self._tensor(decoder_input), teacher_forcing_ratio)
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
            logits = self.model(self._tensor(source), self._tensor(decoder_input), teacher_forcing_ratio=1.0)
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
        encoder_outputs, hidden = self.model.encoder(source)
        mask = source != PAD_ID
        mask[:, 0] = True

        token = torch.tensor([SOS_ID], device=self.device)
        generated: List[int] = []
        for _ in range(max_length):
            logits, hidden = self.model.decoder.step(token, hidden, encoder_outputs, mask)
            next_id = sample_from_logits(logits.squeeze(0).cpu().numpy(), strategy, temperature, top_k)
            if next_id == EOS_ID:
                break
            generated.append(next_id)
            token = torch.tensor([next_id], device=self.device)
        return generated

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)

    def save_weights(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), directory / WEIGHTS_FILE)

    def load_weights(self, directory: Path) -> None:
        state = torch.load(directory / WEIGHTS_FILE, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def state_snapshot(self) -> dict:
        return {key: value.detach().clone() for key, value in self.model.state_dict().items()}

    def restore_snapshot(self, snapshot: dict) -> None:
        self.model.load_state_dict(snapshot)
