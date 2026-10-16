"""Command line front end for the conversational deep learning demo."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from .backends import available_backends
from .chat import chat_loop, show_conversation
from .config import ChatbotConfig, GenerationConfig, ModelConfig, TrainingConfig
from .corpus import DEFAULT_CORPUS_PATH, corpus_statistics, load_pairs, write_corpus
from .trainer import load_chatbot, response_accuracy, save_chatbot, train_chatbot

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "saved_models" / "chatbot"


def _banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def corpus_command(args: argparse.Namespace) -> None:
    path = Path(args.path or DEFAULT_CORPUS_PATH)
    if args.rebuild or not path.is_file():
        write_corpus(path, augment=not args.no_augment)
        print(f"corpus written -> {path}")

    pairs = load_pairs(path)
    _banner(f"Corpus: {path}")
    for name, value in corpus_statistics(pairs).items():
        print(f"  {name:<22} {value:.2f}" if isinstance(value, float) else f"  {name:<22} {value}")
    print("\nFirst few pairs:")
    for prompt, response in pairs[: args.show]:
        print(f"  {prompt!r:<40} -> {response!r}")


def train_command(args: argparse.Namespace) -> None:
    config = ChatbotConfig(
        backend=args.backend,
        corpus_path=args.corpus or "",
        model=ModelConfig(
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.layers,
            dropout=args.dropout,
            use_attention=not args.no_attention,
            architecture=args.architecture,
            num_heads=args.heads,
            feedforward_dim=args.ff_dim,
            num_encoder_layers=args.encoder_layers,
            num_decoder_layers=args.decoder_layers,
        ),
        training=TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            teacher_forcing_ratio=args.teacher_forcing,
            patience=args.patience,
            seed=args.seed,
            log_every=args.log_every,
        ),
        generation=GenerationConfig(strategy=args.strategy, temperature=args.temperature),
    )

    _banner(f"Training the conversation model ({config.backend}, {config.model.architecture})")
    model = train_chatbot(config)
    print("\n" + model.summary())

    pairs = load_pairs(model.config.corpus_path)
    print(f"\nexact reply match on the corpus: {response_accuracy(model, pairs[:120]):.1%}")

    show_conversation(model)

    if args.save:
        target = save_chatbot(model, args.save)
        print(f"model saved -> {target}")
    if args.chat:
        chat_loop(model)


def chat_command(args: argparse.Namespace) -> None:
    model = load_chatbot(args.model)
    print(model.summary())
    if args.samples:
        show_conversation(model)
    else:
        chat_loop(model)


def info_command(args: argparse.Namespace) -> None:
    model = load_chatbot(args.model)
    print(model.summary())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dl-chatbot", description="Lightweight encoder/decoder conversation model (PyTorch or TensorFlow)."
    )
    sub = parser.add_subparsers(dest="command")

    corpus_parser = sub.add_parser("corpus", help="Build or inspect the conversation corpus.")
    corpus_parser.add_argument("--path", help=f"Corpus file (default {DEFAULT_CORPUS_PATH}).")
    corpus_parser.add_argument("--rebuild", action="store_true", help="Regenerate the corpus file.")
    corpus_parser.add_argument("--no-augment", action="store_true", help="Skip the paraphrase augmentation.")
    corpus_parser.add_argument("--show", type=int, default=8, help="How many example pairs to print.")

    train_parser = sub.add_parser("train", help="Train the conversation model.")
    train_parser.add_argument("--backend", choices=["torch", "tensorflow"], default="torch")
    train_parser.add_argument(
        "--architecture", choices=["rnn", "transformer"], default="rnn", help="GRU+attention or transformer."
    )
    train_parser.add_argument("--corpus", help="Corpus JSONL (default: the bundled one).")
    train_parser.add_argument("--epochs", type=int, default=80)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--learning-rate", type=float, default=2e-3)
    train_parser.add_argument("--embedding-dim", type=int, default=128, help="Also the transformer model dim.")
    train_parser.add_argument("--hidden-dim", type=int, default=192, help="RNN only.")
    train_parser.add_argument("--layers", type=int, default=1, help="RNN only.")
    train_parser.add_argument("--heads", type=int, default=4, help="Transformer attention heads.")
    train_parser.add_argument("--ff-dim", type=int, default=512, help="Transformer feed forward width.")
    train_parser.add_argument("--encoder-layers", type=int, default=2, help="Transformer encoder blocks.")
    train_parser.add_argument("--decoder-layers", type=int, default=2, help="Transformer decoder blocks.")
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--no-attention", action="store_true", help="Use a plain context vector instead.")
    train_parser.add_argument("--teacher-forcing", type=float, default=0.9)
    train_parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (0 disables).")
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--log-every", type=int, default=5)
    train_parser.add_argument("--strategy", choices=["greedy", "sampling"], default="greedy")
    train_parser.add_argument("--temperature", type=float, default=0.8)
    train_parser.add_argument("--save", nargs="?", const=str(DEFAULT_MODEL_DIR), help="Directory to save into.")
    train_parser.add_argument("--chat", action="store_true", help="Open the chat loop after training.")

    chat_parser = sub.add_parser("chat", help="Chat with a saved model.")
    chat_parser.add_argument("--model", default=str(DEFAULT_MODEL_DIR))
    chat_parser.add_argument("--samples", action="store_true", help="Print a scripted conversation and exit.")

    info_parser = sub.add_parser("info", help="Show a saved model's details.")
    info_parser.add_argument("--model", default=str(DEFAULT_MODEL_DIR))

    sub.add_parser("backends", help="List the installed deep learning backends.")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0

    if args.command == "backends":
        installed = available_backends()
        _banner("Deep learning backends")
        for backend in ("torch", "tensorflow"):
            print(f"  {backend:<12} {'installed' if backend in installed else 'not installed'}")
        if not installed:
            print("\nInstall one with:  pip install torch    or    pip install tensorflow")
        return 0
    if args.command == "corpus":
        corpus_command(args)
    elif args.command == "train":
        train_command(args)
    elif args.command == "chat":
        chat_command(args)
    elif args.command == "info":
        info_command(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
