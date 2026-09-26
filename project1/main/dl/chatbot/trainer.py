"""The training loop: builds the vocabulary, drives the epochs and keeps the best weights."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from .backends import EpochLog, Seq2SeqBackend, available_backends, create_backend
from .config import ChatbotConfig
from .corpus import DEFAULT_CORPUS_PATH, ensure_corpus, load_pairs
from .dataset import build_dataset, train_validation_split
from .tokenizer import WordTokenizer

CONFIG_FILE = "config.json"
TOKENIZER_FILE = "tokenizer.json"
HISTORY_FILE = "history.json"


@dataclass
class TrainingHistory:
    logs: List[EpochLog] = field(default_factory=list)
    best_epoch: int = 0
    best_loss: float = math.inf
    stopped_early: bool = False

    def to_dict(self) -> dict:
        return {
            "best_epoch": self.best_epoch,
            "best_loss": self.best_loss,
            "stopped_early": self.stopped_early,
            "epochs": [
                {
                    "epoch": log.epoch,
                    "train_loss": log.train_loss,
                    "validation_loss": log.validation_loss,
                    "perplexity": log.perplexity,
                    "seconds": log.seconds,
                }
                for log in self.logs
            ],
        }


@dataclass
class ConversationModel:
    """A trained backend plus the tokenizer needed to talk to it."""

    backend: Seq2SeqBackend
    tokenizer: WordTokenizer
    config: ChatbotConfig
    history: TrainingHistory = field(default_factory=TrainingHistory)

    def reply(
        self,
        message: str,
        strategy: Optional[str] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> str:
        generation = self.config.generation
        source = self.tokenizer.encode(message, max_length=self.config.model.max_source_length)
        if not source:
            return "Could you say that with a few more words?"
        padded = source + [0] * (self.config.model.max_source_length - len(source))
        token_ids = self.backend.generate(
            padded,
            max_length=max_length or generation.max_length,
            strategy=strategy or generation.strategy,
            temperature=generation.temperature if temperature is None else temperature,
            top_k=generation.top_k if top_k is None else top_k,
        )
        text = self.tokenizer.decode(token_ids)
        return text or "I am not sure how to answer that."

    def summary(self) -> str:
        model = self.config.model
        if model.architecture == "transformer":
            shape = [
                f"model dim        : {model.embedding_dim}",
                f"heads/ff dim     : {model.num_heads} / {model.feedforward_dim}",
                f"enc/dec blocks   : {model.num_encoder_layers} / {model.num_decoder_layers}",
            ]
        else:
            shape = [
                f"embedding/hidden : {model.embedding_dim} / {model.hidden_dim}",
                f"layers/attention : {model.num_layers} / {'yes' if model.use_attention else 'no'}",
            ]
        return "\n".join(
            [
                "Conversation model",
                "-" * 60,
                f"backend          : {self.backend.name}",
                f"architecture     : {model.architecture}",
                f"vocabulary       : {self.tokenizer.vocab_size} tokens",
                *shape,
                f"parameters       : {self.backend.parameter_count():,}",
                f"best epoch       : {self.history.best_epoch} (loss {self.history.best_loss:.4f})",
                f"corpus           : {self.config.corpus_path}",
            ]
        )


def train_chatbot(
    config: Optional[ChatbotConfig] = None,
    corpus_path: Optional[str] = None,
    verbose: bool = True,
    on_epoch_end: Optional[Callable[[EpochLog], None]] = None,
) -> ConversationModel:
    config = config or ChatbotConfig()
    source_path = Path(corpus_path or config.corpus_path or ensure_corpus(DEFAULT_CORPUS_PATH))
    config.corpus_path = str(source_path)

    if config.backend not in available_backends():
        installed = available_backends()
        raise RuntimeError(
            f"Backend '{config.backend}' is not installed. "
            f"Available: {installed or 'none'}. Install with: pip install {config.backend}"
        )

    pairs = load_pairs(source_path)
    tokenizer = WordTokenizer().fit(
        [text for pair in pairs for text in pair],
        casing_texts=[response for _, response in pairs],
    )
    dataset = build_dataset(pairs, tokenizer, config.model.max_source_length, config.model.max_target_length)
    train_data, validation_data = train_validation_split(
        dataset, config.training.validation_fraction, config.training.seed
    )

    backend = create_backend(config.backend, tokenizer.vocab_size, config.model, config.training.seed)
    backend.build()
    backend.set_learning_rate(config.training.learning_rate)

    if verbose:
        print(
            f"backend={backend.name}  pairs={len(pairs)}  vocab={tokenizer.vocab_size}  "
            f"train={len(train_data)}  val={len(validation_data)}  "
            f"parameters={backend.parameter_count():,}"
        )
    history = TrainingHistory()
    best_snapshot = None
    epochs_without_improvement = 0

    for epoch in range(1, config.training.epochs + 1):
        started = time.perf_counter()
        train_loss = backend.train_epoch(
            train_data,
            config.training.batch_size,
            config.training.teacher_forcing_ratio,
            config.training.gradient_clip,
            seed=config.training.seed + epoch,
        )
        validation_loss = backend.evaluate(validation_data, config.training.batch_size)
        monitored = train_loss if math.isnan(validation_loss) else validation_loss

        log = EpochLog(
            epoch=epoch,
            train_loss=train_loss,
            validation_loss=None if math.isnan(validation_loss) else validation_loss,
            perplexity=math.exp(min(monitored, 20.0)),
            seconds=time.perf_counter() - started,
        )
        history.logs.append(log)
        if on_epoch_end:
            on_epoch_end(log)
        if verbose and (epoch % max(config.training.log_every, 1) == 0 or epoch == 1):
            print(log.format())

        if monitored < history.best_loss - 1e-4:
            history.best_loss = monitored
            history.best_epoch = epoch
            epochs_without_improvement = 0
            try:
                best_snapshot = backend.state_snapshot()
            except NotImplementedError:
                best_snapshot = None
        else:
            epochs_without_improvement += 1
            if config.training.patience and epochs_without_improvement >= config.training.patience:
                history.stopped_early = True
                if verbose:
                    print(f"early stop at epoch {epoch} (no improvement for {config.training.patience} epochs)")
                break

    if best_snapshot is not None:
        backend.restore_snapshot(best_snapshot)

    return ConversationModel(backend, tokenizer, config, history)


def response_accuracy(model: ConversationModel, pairs: Sequence[Tuple[str, str]]) -> float:
    """Share of prompts whose greedy reply exactly matches the expected response."""
    if not pairs:
        return 0.0
    matches = sum(
        1
        for prompt, expected in pairs
        if model.reply(prompt, strategy="greedy").strip().lower() == expected.strip().lower()
    )
    return matches / len(pairs)


def save_chatbot(model: ConversationModel, directory: str | Path) -> Path:
    target = Path(directory).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    model.backend.save_weights(target)
    model.tokenizer.save(target / TOKENIZER_FILE)
    (target / CONFIG_FILE).write_text(json.dumps(model.config.to_dict(), indent=2), encoding="utf-8")
    (target / HISTORY_FILE).write_text(json.dumps(model.history.to_dict(), indent=2), encoding="utf-8")
    return target


def load_chatbot(directory: str | Path) -> ConversationModel:
    source = Path(directory).expanduser()
    if not (source / CONFIG_FILE).is_file():
        raise FileNotFoundError(f"No saved model in '{source}' (missing {CONFIG_FILE}).")

    config = ChatbotConfig.from_dict(json.loads((source / CONFIG_FILE).read_text(encoding="utf-8")))
    tokenizer = WordTokenizer.load(source / TOKENIZER_FILE)
    backend = create_backend(config.backend, tokenizer.vocab_size, config.model, config.training.seed)
    backend.build()
    backend.load_weights(source)

    history = TrainingHistory()
    history_file = source / HISTORY_FILE
    if history_file.is_file():
        payload = json.loads(history_file.read_text(encoding="utf-8"))
        history.best_epoch = payload.get("best_epoch", 0)
        history.best_loss = payload.get("best_loss", math.inf)
        history.stopped_early = payload.get("stopped_early", False)
    return ConversationModel(backend, tokenizer, config, history)
