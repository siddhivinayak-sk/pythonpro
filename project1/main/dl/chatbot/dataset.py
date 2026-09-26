"""Turns prompt/response pairs into padded id matrices and mini-batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Sequence, Tuple

import numpy as np

from .tokenizer import EOS_ID, PAD_ID, SOS_ID, WordTokenizer


@dataclass
class ConversationDataset:
    """Encoder inputs plus the shifted decoder input/target pair used by teacher forcing."""

    encoder_inputs: np.ndarray  # (n, source_length)
    decoder_inputs: np.ndarray  # (n, target_length)  <sos> w1 w2 ...
    decoder_targets: np.ndarray  # (n, target_length)  w1 w2 ... <eos>

    def __len__(self) -> int:
        return int(self.encoder_inputs.shape[0])

    @property
    def source_length(self) -> int:
        return int(self.encoder_inputs.shape[1])

    @property
    def target_length(self) -> int:
        return int(self.decoder_inputs.shape[1])

    def subset(self, indices: Sequence[int]) -> "ConversationDataset":
        index = np.asarray(indices, dtype=np.int64)
        return ConversationDataset(
            self.encoder_inputs[index],
            self.decoder_inputs[index],
            self.decoder_targets[index],
        )


def _pad(sequence: List[int], length: int) -> List[int]:
    return sequence[:length] + [PAD_ID] * max(0, length - len(sequence))


def build_dataset(
    pairs: Sequence[Tuple[str, str]],
    tokenizer: WordTokenizer,
    max_source_length: int,
    max_target_length: int,
) -> ConversationDataset:
    encoder_rows, decoder_input_rows, decoder_target_rows = [], [], []

    for prompt, response in pairs:
        source = tokenizer.encode(prompt, max_length=max_source_length)
        target = tokenizer.encode(response, max_length=max_target_length - 1)
        encoder_rows.append(_pad(source, max_source_length))
        decoder_input_rows.append(_pad([SOS_ID] + target, max_target_length))
        decoder_target_rows.append(_pad(target + [EOS_ID], max_target_length))

    return ConversationDataset(
        np.asarray(encoder_rows, dtype=np.int64),
        np.asarray(decoder_input_rows, dtype=np.int64),
        np.asarray(decoder_target_rows, dtype=np.int64),
    )


def train_validation_split(
    dataset: ConversationDataset,
    validation_fraction: float,
    seed: int = 42,
) -> Tuple[ConversationDataset, ConversationDataset]:
    count = len(dataset)
    indices = np.arange(count)
    np.random.default_rng(seed).shuffle(indices)

    validation_size = int(round(count * validation_fraction))
    validation_size = min(max(validation_size, 1 if validation_fraction > 0 else 0), count - 1)
    if validation_size == 0:
        return dataset, dataset.subset([])
    return dataset.subset(indices[validation_size:]), dataset.subset(indices[:validation_size])


def iterate_batches(
    dataset: ConversationDataset,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 0,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    count = len(dataset)
    order = np.arange(count)
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for start in range(0, count, batch_size):
        index = order[start : start + batch_size]
        yield (
            dataset.encoder_inputs[index],
            dataset.decoder_inputs[index],
            dataset.decoder_targets[index],
        )
