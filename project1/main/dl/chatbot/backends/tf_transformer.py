"""TensorFlow/Keras transformer encoder/decoder.

Same architecture as the PyTorch transformer, but built on `keras.layers.MultiHeadAttention`
so the two files show the low level and the high level way of writing the same model.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import List

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf

tf.get_logger().setLevel("ERROR")
# The first decoding step attends over a single token, which Keras flags as a size-1 softmax.
warnings.filterwarnings("ignore", message=".*softmax over axis.*")

from ..config import ModelConfig
from ..dataset import ConversationDataset, iterate_batches
from ..tokenizer import EOS_ID, PAD_ID, SOS_ID
from .base import Seq2SeqBackend, sample_from_logits
from .tf_backend import masked_loss

WEIGHTS_FILE = "model.weights.h5"


def positional_encoding(max_length: int, model_dim: int) -> tf.Tensor:
    """The classic sinusoidal position signal, since attention itself is order blind."""
    position = np.arange(max_length)[:, None]
    divisor = np.exp(np.arange(0, model_dim, 2) * (-np.log(10000.0) / model_dim))
    encoding = np.zeros((max_length, model_dim), dtype=np.float32)
    encoding[:, 0::2] = np.sin(position * divisor)
    encoding[:, 1::2] = np.cos(position * divisor)
    return tf.constant(encoding[None, ...])


class EncoderLayer(tf.keras.layers.Layer):
    """Self-attention over the prompt, then a position-wise feed forward block."""

    def __init__(self, config: ModelConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.self_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=config.num_heads, key_dim=config.embedding_dim // config.num_heads, dropout=config.dropout
        )
        self.feed_forward = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(config.feedforward_dim, activation="relu"),
                tf.keras.layers.Dropout(config.dropout),
                tf.keras.layers.Dense(config.embedding_dim),
            ]
        )
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(config.dropout)

    def call(self, x, source_mask, training: bool = False):
        attended = self.self_attention(x, x, attention_mask=source_mask, training=training)
        x = self.norm1(x + self.dropout(attended, training=training))
        return self.norm2(x + self.dropout(self.feed_forward(x, training=training), training=training))


class DecoderLayer(tf.keras.layers.Layer):
    """Causal self-attention over the reply, cross-attention to the prompt, then feed forward."""

    def __init__(self, config: ModelConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        key_dim = config.embedding_dim // config.num_heads
        self.self_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=config.num_heads, key_dim=key_dim, dropout=config.dropout
        )
        self.cross_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=config.num_heads, key_dim=key_dim, dropout=config.dropout
        )
        self.feed_forward = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(config.feedforward_dim, activation="relu"),
                tf.keras.layers.Dropout(config.dropout),
                tf.keras.layers.Dense(config.embedding_dim),
            ]
        )
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(config.dropout)

    def call(self, x, memory, target_mask, source_mask, training: bool = False):
        attended = self.self_attention(x, x, attention_mask=target_mask, training=training)
        x = self.norm1(x + self.dropout(attended, training=training))
        attended = self.cross_attention(x, memory, attention_mask=source_mask, training=training)
        x = self.norm2(x + self.dropout(attended, training=training))
        return self.norm3(x + self.dropout(self.feed_forward(x, training=training), training=training))


def padding_mask(source, query_length):
    """(batch, query_length, source_length) - True where the key token is real."""
    mask = tf.not_equal(source, PAD_ID)
    mask = tf.concat([tf.ones_like(mask[:, :1]), mask[:, 1:]], axis=1)
    return tf.repeat(mask[:, None, :], repeats=query_length, axis=1)


def causal_mask(target):
    """(batch, length, length) - position t may only attend to positions <= t."""
    length = tf.shape(target)[1]
    look_ahead = tf.linalg.band_part(tf.ones((length, length), dtype=tf.bool), -1, 0)
    valid = tf.not_equal(target, PAD_ID)
    valid = tf.concat([tf.ones_like(valid[:, :1]), valid[:, 1:]], axis=1)
    return tf.logical_and(look_ahead[None, :, :], valid[:, None, :])


class TransformerSeq2Seq(tf.keras.Model):
    def __init__(self, vocab_size: int, config: ModelConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.scale = tf.math.sqrt(tf.cast(config.embedding_dim, tf.float32))
        # Prompts and replies are the same language, so the embedding table is shared.
        self.embedding = tf.keras.layers.Embedding(vocab_size, config.embedding_dim)
        self.positional = positional_encoding(512, config.embedding_dim)
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.encoder_layers = [EncoderLayer(config) for _ in range(config.num_encoder_layers)]
        self.decoder_layers = [DecoderLayer(config) for _ in range(config.num_decoder_layers)]
        self.output_layer = tf.keras.layers.Dense(vocab_size)

    def build(self, input_shape):
        # Sub-layers are built lazily on the first call; declaring build keeps Keras quiet.
        super().build(input_shape)

    def _embed(self, tokens, training: bool):
        length = tf.shape(tokens)[1]
        embedded = self.embedding(tokens) * self.scale + self.positional[:, :length]
        return self.dropout(embedded, training=training)

    def encode(self, source, training: bool = False):
        source_length = tf.shape(source)[1]
        mask = padding_mask(source, source_length)
        x = self._embed(source, training)
        for layer in self.encoder_layers:
            x = layer(x, mask, training=training)
        return x

    def decode(self, target_inputs, memory, source, training: bool = False):
        target_length = tf.shape(target_inputs)[1]
        cross_mask = padding_mask(source, target_length)
        x = self._embed(target_inputs, training)
        for layer in self.decoder_layers:
            x = layer(x, memory, causal_mask(target_inputs), cross_mask, training=training)
        return self.output_layer(x)

    def call(self, inputs, training: bool = False):
        source, target_inputs = inputs
        return self.decode(target_inputs, self.encode(source, training), source, training)


class TensorFlowTransformer(Seq2SeqBackend):
    name = "tensorflow-transformer"

    def build(self) -> None:
        tf.keras.utils.set_random_seed(self.seed)
        self.model = TransformerSeq2Seq(self.vocab_size, self.config)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_2=0.98, epsilon=1e-9)
        self.clip_norm = 1.0
        dummy_source = tf.zeros((1, self.config.max_source_length), dtype=tf.int64)
        dummy_target = tf.zeros((1, self.config.max_target_length), dtype=tf.int64)
        self.model((dummy_source, dummy_target), training=False)

    def set_learning_rate(self, learning_rate: float) -> None:
        self.optimizer.learning_rate.assign(learning_rate)

    @tf.function(reduce_retracing=True)
    def _train_batch(self, source, decoder_input, target):
        with tf.GradientTape() as tape:
            logits = self.model((source, decoder_input), training=True)
            loss = masked_loss(target, logits)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, self.clip_norm)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    @tf.function(reduce_retracing=True)
    def _evaluate_batch(self, source, decoder_input, target):
        return masked_loss(target, self.model((source, decoder_input), training=False))

    def train_epoch(
        self,
        dataset: ConversationDataset,
        batch_size: int,
        teacher_forcing_ratio: float,
        gradient_clip: float,
        seed: int,
    ) -> float:
        # teacher_forcing_ratio is unused: a transformer sees the whole shifted reply at once.
        self.clip_norm = gradient_clip if gradient_clip > 0 else 1e9
        total_loss, batches = 0.0, 0
        for source, decoder_input, target in iterate_batches(dataset, batch_size, shuffle=True, seed=seed):
            loss = self._train_batch(
                tf.convert_to_tensor(source),
                tf.convert_to_tensor(decoder_input),
                tf.convert_to_tensor(target),
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
        memory = self.model.encode(source, training=False)

        tokens = [SOS_ID]
        generated: List[int] = []
        for _ in range(max_length):
            decoder_input = tf.constant([tokens], dtype=tf.int64)
            logits = self.model.decode(decoder_input, memory, source, training=False)
            next_id = sample_from_logits(logits.numpy()[0, -1], strategy, temperature, top_k)
            if next_id == EOS_ID:
                break
            generated.append(next_id)
            tokens.append(next_id)
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
