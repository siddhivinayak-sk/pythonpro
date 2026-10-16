"""TensorFlow/Keras GRU encoder / attention decoder and its training step.

Mirrors the PyTorch backend layer for layer so the two can be compared directly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf

tf.get_logger().setLevel("ERROR")

from ..config import ModelConfig
from ..dataset import ConversationDataset, iterate_batches
from ..tokenizer import EOS_ID, PAD_ID, SOS_ID
from .base import Seq2SeqBackend, sample_from_logits

WEIGHTS_FILE = "model.weights.h5"


class Encoder(tf.keras.layers.Layer):
    """Embeds the prompt and runs stacked GRUs, keeping every step for attention."""

    def __init__(self, vocab_size: int, config: ModelConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.embedding = tf.keras.layers.Embedding(vocab_size, config.embedding_dim)
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.grus = [
            tf.keras.layers.GRU(config.hidden_dim, return_sequences=True, return_state=True)
            for _ in range(config.num_layers)
        ]

    def call(self, source, training: bool = False):
        outputs = self.dropout(self.embedding(source), training=training)
        states = []
        for gru in self.grus:
            outputs, state = gru(outputs, training=training)
            states.append(state)
        return outputs, states


class LuongAttention(tf.keras.layers.Layer):
    """score(h_dec, h_enc) = h_dec^T W h_enc, masked over padding."""

    def __init__(self, hidden_dim: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.project = tf.keras.layers.Dense(hidden_dim, use_bias=False)

    def call(self, decoder_state, encoder_outputs, mask):
        scores = tf.squeeze(
            tf.matmul(encoder_outputs, tf.expand_dims(self.project(decoder_state), axis=2)), axis=2
        )
        scores = tf.where(mask, scores, tf.fill(tf.shape(scores), scores.dtype.min))
        weights = tf.expand_dims(tf.nn.softmax(scores, axis=1), axis=1)
        return tf.squeeze(tf.matmul(weights, encoder_outputs), axis=1)


class Decoder(tf.keras.layers.Layer):
    """Generates the reply one token at a time, optionally attending to the encoder."""

    def __init__(self, vocab_size: int, config: ModelConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.use_attention = config.use_attention
        self.embedding = tf.keras.layers.Embedding(vocab_size, config.embedding_dim)
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.cells = [tf.keras.layers.GRUCell(config.hidden_dim) for _ in range(config.num_layers)]
        self.attention = LuongAttention(config.hidden_dim) if config.use_attention else None
        self.output_layer = tf.keras.layers.Dense(vocab_size)

    def step(self, token, states, encoder_outputs, mask, training: bool = False):
        hidden = self.dropout(self.embedding(token), training=training)
        new_states = []
        for cell, state in zip(self.cells, states):
            hidden, [state] = cell(hidden, [state], training=training)
            new_states.append(state)
        if self.attention is not None:
            context = self.attention(hidden, encoder_outputs, mask)
            hidden = tf.concat([hidden, context], axis=1)
        return self.output_layer(hidden), new_states


class Seq2Seq(tf.keras.Model):
    def __init__(self, vocab_size: int, config: ModelConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.encoder = Encoder(vocab_size, config)
        self.decoder = Decoder(vocab_size, config)

    def build(self, input_shape):
        # Sub-layers are built lazily on the first call; declaring build keeps Keras quiet.
        super().build(input_shape)

    def call(self, inputs, training: bool = False, teacher_forcing_ratio: float = 1.0):
        source, decoder_inputs = inputs
        encoder_outputs, states = self.encoder(source, training=training)
        mask = tf.not_equal(source, PAD_ID)
        # A prompt of only padding would make the masked softmax NaN.
        mask = tf.concat([tf.ones_like(mask[:, :1]), mask[:, 1:]], axis=1)

        steps = decoder_inputs.shape[1]
        token = decoder_inputs[:, 0]
        collected = []
        for position in range(steps):
            logits, states = self.decoder.step(token, states, encoder_outputs, mask, training=training)
            collected.append(logits)
            if position + 1 < steps:
                use_teacher = tf.random.uniform([]) < teacher_forcing_ratio
                token = tf.cond(
                    use_teacher,
                    lambda: decoder_inputs[:, position + 1],
                    lambda: tf.cast(tf.argmax(logits, axis=1), decoder_inputs.dtype),
                )
        return tf.stack(collected, axis=1)


def masked_loss(targets, logits) -> tf.Tensor:
    losses = tf.keras.losses.sparse_categorical_crossentropy(targets, logits, from_logits=True)
    mask = tf.cast(tf.not_equal(targets, PAD_ID), losses.dtype)
    return tf.reduce_sum(losses * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)


class TensorFlowSeq2Seq(Seq2SeqBackend):
    name = "tensorflow"

    def build(self) -> None:
        tf.keras.utils.set_random_seed(self.seed)
        self.model = Seq2Seq(self.vocab_size, self.config)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=2e-3)
        self.clip_norm = 1.0
        # Force variable creation so weights can be saved/loaded before any training.
        dummy_source = tf.zeros((1, self.config.max_source_length), dtype=tf.int64)
        dummy_target = tf.zeros((1, self.config.max_target_length), dtype=tf.int64)
        self.model((dummy_source, dummy_target), training=False)

    def set_learning_rate(self, learning_rate: float) -> None:
        self.optimizer.learning_rate.assign(learning_rate)

    @tf.function(reduce_retracing=True)
    def _train_batch(self, source, decoder_input, target, teacher_forcing_ratio):
        with tf.GradientTape() as tape:
            logits = self.model(
                (source, decoder_input), training=True, teacher_forcing_ratio=teacher_forcing_ratio
            )
            loss = masked_loss(target, logits)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, self.clip_norm)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    @tf.function(reduce_retracing=True)
    def _evaluate_batch(self, source, decoder_input, target):
        logits = self.model((source, decoder_input), training=False, teacher_forcing_ratio=1.0)
        return masked_loss(target, logits)

    def train_epoch(
        self,
        dataset: ConversationDataset,
        batch_size: int,
        teacher_forcing_ratio: float,
        gradient_clip: float,
        seed: int,
    ) -> float:
        self.clip_norm = gradient_clip if gradient_clip > 0 else 1e9
        ratio = tf.constant(teacher_forcing_ratio, dtype=tf.float32)
        total_loss, batches = 0.0, 0
        for source, decoder_input, target in iterate_batches(dataset, batch_size, shuffle=True, seed=seed):
            loss = self._train_batch(
                tf.convert_to_tensor(source),
                tf.convert_to_tensor(decoder_input),
                tf.convert_to_tensor(target),
                ratio,
            )
            total_loss += float(loss.numpy())
            batches += 1
        return total_loss / max(batches, 1)

    def evaluate(self, dataset: ConversationDataset, batch_size: int) -> float:
        if len(dataset) == 0:
            return float("nan")
        total_loss, batches = 0.0, 0
        for source, decoder_input, target in iterate_batches(dataset, batch_size, shuffle=False):
            loss = self._evaluate_batch(
                tf.convert_to_tensor(source),
                tf.convert_to_tensor(decoder_input),
                tf.convert_to_tensor(target),
            )
            total_loss += float(loss.numpy())
            batches += 1
        return total_loss / max(batches, 1)

    def generate(
        self,
        source_ids: np.ndarray,
        max_length: int,
        strategy: str = "greedy",
        temperature: float = 0.8,
        top_k: int = 5,
    ) -> List[int]:
        source = tf.convert_to_tensor(np.asarray(source_ids, dtype=np.int64).reshape(1, -1))
        encoder_outputs, states = self.model.encoder(source, training=False)
        mask = tf.not_equal(source, PAD_ID)
        mask = tf.concat([tf.ones_like(mask[:, :1]), mask[:, 1:]], axis=1)

        token = tf.constant([SOS_ID], dtype=tf.int64)
        generated: List[int] = []
        for _ in range(max_length):
            logits, states = self.model.decoder.step(token, states, encoder_outputs, mask, training=False)
            next_id = sample_from_logits(logits.numpy()[0], strategy, temperature, top_k)
            if next_id == EOS_ID:
                break
            generated.append(next_id)
            token = tf.constant([next_id], dtype=tf.int64)
        return generated

    def parameter_count(self) -> int:
        return int(sum(int(np.prod(variable.shape)) for variable in self.model.trainable_variables))

    def save_weights(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(str(directory / WEIGHTS_FILE))

    def load_weights(self, directory: Path) -> None:
        self.model.load_weights(str(directory / WEIGHTS_FILE))

    def state_snapshot(self) -> List[np.ndarray]:
        return [np.array(weight) for weight in self.model.get_weights()]

    def restore_snapshot(self, snapshot: List[np.ndarray]) -> None:
        self.model.set_weights(snapshot)
