"""End-to-end demonstration: corpus -> training loop -> conversation.

Run with:  python -m main.dl.chatbot.demo                    (fastest available backend, GRU)
           python -m main.dl.chatbot.demo transformer
           python -m main.dl.chatbot.demo tensorflow transformer 60
           python -m main.dl.chatbot.demo both all
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

from .backends import available_backends
from .chat import show_conversation
from .config import ChatbotConfig, GenerationConfig, ModelConfig, TrainingConfig
from .corpus import DEFAULT_CORPUS_PATH, corpus_statistics, ensure_corpus, load_pairs
from .trainer import load_chatbot, response_accuracy, save_chatbot, train_chatbot

MODEL_ROOT = Path(__file__).resolve().parent.parent / "saved_models"


def _header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def demo_corpus() -> None:
    _header("1. The corpus")
    path = ensure_corpus(DEFAULT_CORPUS_PATH)
    pairs = load_pairs(path)
    print(f"file: {path}")
    for name, value in corpus_statistics(pairs).items():
        print(f"  {name:<22} {value:.2f}" if isinstance(value, float) else f"  {name:<22} {value}")
    print("\nexamples:")
    for prompt, response in pairs[:5]:
        print(f"  {prompt!r:<38} -> {response!r}")


def demo_backend(backend: str, architecture: str = "rnn", epochs: int = 80) -> None:
    _header(f"2. Training the encoder/decoder with {backend} ({architecture})")
    config = ChatbotConfig(
        backend=backend,
        model=ModelConfig(
            embedding_dim=128,
            hidden_dim=192,
            num_layers=1,
            use_attention=True,
            architecture=architecture,
            num_heads=4,
            feedforward_dim=512,
            num_encoder_layers=2,
            num_decoder_layers=2,
        ),
        training=TrainingConfig(
            epochs=epochs,
            batch_size=32,
            learning_rate=2e-3 if architecture == "rnn" else 1e-3,
            patience=15,
            log_every=5,
        ),
        generation=GenerationConfig(strategy="greedy", max_length=24),
    )
    model = train_chatbot(config)

    print("\n" + model.summary())
    pairs = load_pairs(model.config.corpus_path)
    print(f"\nexact reply match on 120 corpus prompts: {response_accuracy(model, pairs[:120]):.1%}")

    show_conversation(model)

    _header(f"3. Saving and reloading the {backend} {architecture} model")
    directory = save_chatbot(model, MODEL_ROOT / f"chatbot_{backend}_{architecture}")
    print(f"saved -> {directory}")
    reloaded = load_chatbot(directory)
    for prompt in ["hello", "what is machine learning", "thanks", "goodbye"]:
        print(f"you : {prompt}\nbot : {reloaded.reply(prompt)}")

    print("\nSame prompt with sampling (temperature 1.0) to show non-greedy decoding:")
    for _ in range(3):
        print(f"  bot : {reloaded.reply('tell me a joke', strategy='sampling', temperature=1.0)}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    installed = available_backends()
    if not installed:
        print("No deep learning backend installed. Run: pip install torch   (or tensorflow)")
        return 1

    backends = [installed[0]]
    architectures = ["rnn"]
    epochs = 80
    for token in argv:
        if token in ("torch", "tensorflow"):
            backends = [token]
        elif token == "both":
            backends = list(installed)
        elif token in ("rnn", "transformer"):
            architectures = [token]
        elif token == "all":
            architectures = ["rnn", "transformer"]
        elif token.isdigit():
            epochs = int(token)
        else:
            print(f"Unknown argument '{token}'. Use: torch|tensorflow|both rnn|transformer|all <epochs>")
            return 1

    demo_corpus()
    for backend in backends:
        if backend not in installed:
            print(f"\nSkipping '{backend}': not installed.")
            continue
        for architecture in architectures:
            demo_backend(backend, architecture, epochs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
