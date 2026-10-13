# pip install prompty[azure] python-dotenv
#
# Prompty - prompt-as-code files (.prompty) with Foundry tracing support.
#
# A .prompty file is a Markdown-with-front-matter template that captures
# model config, inputs, sample data, and the prompt itself, so prompts
# live next to your code and are versioned like any other artifact.
#
# This script writes a small .prompty file, then executes it with the
# `prompty` runtime. Foundry portal can also import .prompty files.

import os
from pathlib import Path
import prompty
import prompty.azure           # registers the azure_openai invoker
from prompty.tracer import trace, Tracer, console_tracer
from dotenv import load_dotenv

load_dotenv()

# 1. Wire up a tracer so we can see calls (Foundry has a tracing exporter too)
Tracer.add("console", console_tracer)

# 2. Write a sample .prompty file next to this script
prompty_path = Path(__file__).with_name("summarize.prompty")
prompty_path.write_text(
    """---
name: Summarize
description: Summarize a piece of text in N bullets.
authors:
  - your-name
model:
  api: chat
  configuration:
    type: azure_openai
    azure_endpoint: ${env:AZURE_OPENAI_ENDPOINT}
    azure_deployment: ${env:AZURE_OPENAI_DEPLOYMENT}
    api_version: ${env:AZURE_OPENAI_API_VERSION}
  parameters:
    max_tokens: 400
    temperature: 0.3
inputs:
  topic:
    type: string
  bullets:
    type: integer
    default: 3
sample:
  topic: "The Apollo program"
  bullets: 4
---

system:
You are a technical writer. Summarize the topic in exactly {{bullets}} short bullets.

user:
Topic: {{topic}}
""",
    encoding="utf-8",
)

# 3. Execute the prompty as a function
@trace
def run(topic: str, bullets: int = 3) -> str:
    return prompty.execute(
        str(prompty_path),
        inputs={"topic": topic, "bullets": bullets},
    )

if __name__ == "__main__":
    output = run(topic="Azure AI Foundry", bullets=5)
    print("\n--- Model output ---")
    print(output)
