"""Foundry Local (SDK v2) - download a small model locally and query it with the OpenAI SDK.

What this script does
---------------------
1. Initializes the Foundry Local SDK (``foundry_local_sdk``, v2.x) in-process.
2. Downloads and registers the execution providers (ORT/WinML EPs) available on this machine.
   This runs on every start: the binaries are cached, but registration is per-process.
3. Downloads a *small* chat model from the public catalog (default ``qwen2.5-0.5b``) and loads it.
4. Starts Foundry Local's built-in OpenAI-compatible HTTP server.
5. Talks to that endpoint with the plain ``openai`` Python client (``base_url = <url>/v1``).
6. Unloads the model and stops the web service on exit.

Everything runs on-device: no Azure resource, no API key, no network calls after the
model is cached.

Setup
-----
    python -m pip install --upgrade foundry-local-sdk openai

Requires Python 3.11-3.14 on Windows x64, Linux x64, or macOS arm64.

Usage
-----
    python foundry_local.py                            # one-shot streaming query
    python foundry_local.py --prompt "What is ONNX?"    # custom prompt
    python foundry_local.py --no-stream                # single non-streaming response
    python foundry_local.py --chat                     # interactive multi-turn REPL
    python foundry_local.py --list-small               # show small catalog models and exit
    python foundry_local.py --model phi-4-mini         # pick another catalog alias

References
----------
- SDK README / API: https://pypi.org/project/foundry-local-sdk/
- Inference SDK integration: https://learn.microsoft.com/azure/foundry-local/how-to/how-to-integrate-with-inference-sdks
(Reference material summarized/rephrased for licensing compliance.)
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

try:
    import openai
    from foundry_local_sdk import Configuration, FoundryLocalManager, LogLevel
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        f"Missing dependency: {exc.name}.\n"
        "Install the requirements first:\n"
        "    python -m pip install --upgrade foundry-local-sdk openai"
    ) from exc


APP_NAME = "pythonpro-foundry-local"

# Small, CPU-friendly chat models from the public Foundry Local catalog.
# ~0.4-1 GB on disk, which keeps the first-run download short.
SMALL_MODEL_ALIASES: tuple[str, ...] = (
    "qwen2.5-0.5b",
    "qwen2.5-1.5b",
    "deepseek-r1-1.5b",
    "phi-3.5-mini",
    "phi-4-mini",
)

DEFAULT_MODEL_ALIAS = "qwen2.5-0.5b"
SYSTEM_PROMPT = "You are a concise, helpful assistant running fully on the user's machine."


# --------------------------------------------------------------------------------------
# Control plane: Foundry Local SDK (discover -> download -> load -> serve)
# --------------------------------------------------------------------------------------
def init_manager(verbose: bool = False) -> Any:
    """Initialize the singleton FoundryLocalManager and return it."""
    config = Configuration(
        app_name=APP_NAME,
        log_level=LogLevel.INFORMATION if verbose else LogLevel.WARNING,
    )
    FoundryLocalManager.initialize(config)
    return FoundryLocalManager.instance


def register_execution_providers(manager: Any) -> None:
    """Download + register the execution providers available on this hardware.

    Registration lives in the current process only: it has to run on every start, even
    though the downloaded EP binaries are cached on disk (so later runs are quick).
    Skipping it makes ``model.load()`` fail for any variant that needs a non-default EP,
    e.g. ``...-generic-gpu`` needs ``WebGpuExecutionProvider``.
    """
    state = {"current": ""}

    def on_progress(ep_name: str, percent: float) -> None:
        if ep_name != state["current"]:
            if state["current"]:
                print()
            state["current"] = ep_name
        print(f"\r  {ep_name:<32} {percent:5.1f}%", end="", flush=True)

    print("Resolving execution providers...")
    result = manager.download_and_register_eps(progress_callback=on_progress)
    if state["current"]:
        print()
    print(f"  execution providers ready: success={result.success} status={result.status}")


def list_small_models(manager: Any) -> None:
    """Print the catalog entries that match our 'small model' shortlist."""
    print("Small chat models in the public catalog:")
    for alias in SMALL_MODEL_ALIASES:
        try:
            model = manager.catalog.get_model(alias)
        except Exception as exc:  # offline / catalog error
            print(f"  - {alias:<18} lookup failed ({exc})")
            continue
        if model is None:  # alias not offered for this hardware/catalog revision
            print(f"  - {alias:<18} not available in this catalog")
            continue
        print(
            f"  - {alias:<18} id={model.id:<48} "
            f"cached={bool(model.is_cached)} ctx={model.context_length}"
        )


def download_and_load(manager: Any, alias: str) -> Any:
    """Fetch the model into the local cache (if needed) and load it into memory."""
    model = manager.catalog.get_model(alias)
    if model is None:
        raise SystemExit(
            f"Model alias '{alias}' was not found in the catalog for this machine.\n"
            "Run with --list-small to see the available small models."
        )
    print(f"Model: alias={model.alias} id={model.id} cached={bool(model.is_cached)}")

    if not model.is_cached:
        print("Downloading model (first run only)...")
        model.download(
            lambda percent: print(f"\r  {percent:6.2f}%", end="", flush=True)
        )
        print()
    else:
        print("  already cached, skipping download")

    print("Loading model into memory...")
    model.load()
    print("  loaded")
    return model


@contextmanager
def openai_compatible_endpoint(manager: Any) -> Iterator[str]:
    """Start Foundry Local's built-in REST server and yield the OpenAI base URL.

    The server binds to localhost only and needs no credentials, which is why the
    OpenAI client below passes a placeholder API key. Do not expose this port on a
    network interface: there is no authentication in front of it.
    """
    manager.start_web_service()
    try:
        urls = list(manager.urls)
        if not urls:
            raise RuntimeError("Foundry Local web service reported no listening URLs")
        base_url = f"{urls[0].rstrip('/')}/v1"
        print(f"OpenAI-compatible endpoint: {base_url}")
        yield base_url
    finally:
        manager.stop_web_service()
        print("Web service stopped")


# --------------------------------------------------------------------------------------
# Data plane: OpenAI Python client against the local endpoint
# --------------------------------------------------------------------------------------
def make_client(base_url: str) -> openai.OpenAI:
    """OpenAI client pointed at the local Foundry endpoint (API key is a placeholder)."""
    return openai.OpenAI(base_url=base_url, api_key="not-needed-for-foundry-local")


def ask_streaming(client: openai.OpenAI, model_id: str, messages: list[dict[str, str]]) -> str:
    """Stream a chat completion, printing tokens as they arrive. Returns the full text."""
    stream = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=0.2,
        max_tokens=512,
        stream=True,
    )
    chunks: list[str] = []
    for chunk in stream:
        if not chunk.choices:
            continue
        piece = chunk.choices[0].delta.content
        if piece:
            chunks.append(piece)
            print(piece, end="", flush=True)
    print()
    return "".join(chunks)


def ask(client: openai.OpenAI, model_id: str, messages: list[dict[str, str]]) -> str:
    """Single non-streaming chat completion. Returns the assistant text."""
    completion = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=0.2,
        max_tokens=512,
    )
    answer = completion.choices[0].message.content or ""
    print(answer)
    if completion.usage:
        print(
            f"\n[tokens] prompt={completion.usage.prompt_tokens} "
            f"completion={completion.usage.completion_tokens} "
            f"total={completion.usage.total_tokens}"
        )
    return answer


def chat_repl(client: openai.OpenAI, model_id: str) -> None:
    """Interactive multi-turn loop; history is sent with every request."""
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("\nInteractive chat. Type 'exit' or 'quit' (or Ctrl+C) to stop.\n")
    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            return
        messages.append({"role": "user", "content": user_input})
        print("model> ", end="", flush=True)
        reply = ask_streaming(client, model_id, messages)
        messages.append({"role": "assistant", "content": reply})


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small model locally with Foundry Local and query it via the OpenAI SDK.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ALIAS,
        help=f"catalog alias to download and load (default: {DEFAULT_MODEL_ALIAS})",
    )
    parser.add_argument(
        "--prompt",
        default="In two sentences, what is Foundry Local and why run models on-device?",
        help="prompt for the one-shot query",
    )
    parser.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        help="return the whole completion at once instead of streaming tokens",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="start an interactive multi-turn chat instead of a single query",
    )
    parser.add_argument(
        "--list-small",
        action="store_true",
        help="list the small catalog models and exit",
    )
    parser.add_argument("--verbose", action="store_true", help="enable INFORMATION-level SDK logs")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    manager = init_manager(verbose=args.verbose)

    if args.list_small:
        list_small_models(manager)
        return 0

    # Must run every process start - EP registration is not persisted across runs.
    register_execution_providers(manager)

    model = download_and_load(manager, args.model)
    try:
        with openai_compatible_endpoint(manager) as base_url:
            client = make_client(base_url)

            if args.chat:
                chat_repl(client, model.id)
                return 0

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": args.prompt},
            ]
            print(f"\nyou>   {args.prompt}\nmodel> ", end="", flush=True)
            if args.stream:
                ask_streaming(client, model.id, messages)
            else:
                print()
                ask(client, model.id, messages)
    finally:
        model.unload()
        print("Model unloaded")

    return 0


if __name__ == "__main__":
    sys.exit(main())
