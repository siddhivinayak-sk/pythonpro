"""Interactive chat loop over a trained conversation model."""

from __future__ import annotations

from typing import Optional, Sequence

from .trainer import ConversationModel

EXIT_WORDS = {"exit", "quit", ":q", "/exit"}

SAMPLE_PROMPTS = [
    "hello",
    "what is your name",
    "what can you do",
    "what is deep learning",
    "explain pytorch",
    "tell me a joke",
    "i am tired",
    "thanks",
    "bye",
]


def show_conversation(model: ConversationModel, prompts: Optional[Sequence[str]] = None) -> None:
    """Print a scripted conversation - the quickest way to demo the model."""
    print("\nSample conversation")
    print("-" * 60)
    for prompt in prompts or SAMPLE_PROMPTS:
        print(f"you : {prompt}")
        print(f"bot : {model.reply(prompt)}\n")


def chat_loop(model: ConversationModel) -> None:
    print("\n" + "=" * 60)
    print("Chat with the model. Type 'exit' to leave.")
    print("Commands: /greedy, /sample, /temp <value>, /help")
    print("=" * 60)

    strategy = model.config.generation.strategy
    temperature = model.config.generation.temperature

    while True:
        try:
            message = input("\nyou : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")
            return
        if not message:
            continue
        if message.lower() in EXIT_WORDS:
            print("bot : Goodbye!")
            return
        if message.startswith("/"):
            strategy, temperature = _handle_command(message, strategy, temperature)
            continue
        print(f"bot : {model.reply(message, strategy=strategy, temperature=temperature)}")


def _handle_command(message: str, strategy: str, temperature: float):
    parts = message.split()
    command = parts[0].lower()
    if command == "/greedy":
        print("(switched to greedy decoding)")
        return "greedy", temperature
    if command in ("/sample", "/sampling"):
        print(f"(switched to sampling, temperature {temperature})")
        return "sampling", temperature
    if command == "/temp" and len(parts) > 1:
        try:
            temperature = float(parts[1])
            print(f"(temperature set to {temperature})")
        except ValueError:
            print("(usage: /temp 0.8)")
        return strategy, temperature
    print("commands: /greedy, /sample, /temp <value>, /help, exit")
    return strategy, temperature
