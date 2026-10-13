# pip install azure-ai-evaluation python-dotenv
#
# Azure AI Foundry - Evaluations SDK
# Runs quality and safety evaluators against sample AI outputs.
# Same evaluators the Foundry portal uses in the "Evaluate" tab.
#
# Two flavors shown:
#   * Local per-row scoring with individual evaluators
#   * Batch evaluate() over a JSONL dataset with multiple evaluators

import os
import json
import tempfile

from azure.ai.evaluation import (
    RelevanceEvaluator,
    GroundednessEvaluator,
    CoherenceEvaluator,
    FluencyEvaluator,
    SimilarityEvaluator,
    evaluate,
)
from dotenv import load_dotenv

load_dotenv()

# Evaluators that use an LLM-as-judge need an Azure OpenAI model config.
model_config = {
    "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
    "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
    "azure_deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
}

relevance = RelevanceEvaluator(model_config)
groundedness = GroundednessEvaluator(model_config)
coherence = CoherenceEvaluator(model_config)

# --- 1. Score a single row ---------------------------------------------

row = {
    "query": "What is the capital of France?",
    "response": "The capital of France is Paris, a major European city.",
    "context": "France is a country in Western Europe; its capital is Paris.",
    "ground_truth": "Paris",
}

print("Single-row scores:")
print(f"  relevance   : {relevance(**row)}")
print(f"  groundedness: {groundedness(response=row['response'], context=row['context'], query=row['query'])}")
print(f"  coherence   : {coherence(query=row['query'], response=row['response'])}")

# --- 2. Batch evaluate over a JSONL dataset ---------------------------

rows = [
    {
        "query": "What is Azure AI Foundry?",
        "response": "Azure AI Foundry is Microsoft's unified platform for building AI apps and agents.",
        "context": "Azure AI Foundry unifies models, agents, safety, and evaluation on Azure.",
        "ground_truth": "A Microsoft platform for building AI apps and agents.",
    },
    {
        "query": "What is the capital of France?",
        "response": "Berlin is the capital of France.",   # deliberately wrong
        "context": "France's capital is Paris.",
        "ground_truth": "Paris",
    },
]

with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
    dataset_path = f.name

results = evaluate(
    data=dataset_path,
    evaluators={
        "relevance": relevance,
        "groundedness": groundedness,
        "coherence": coherence,
        "fluency": FluencyEvaluator(model_config),
        "similarity": SimilarityEvaluator(model_config),
    },
    # Uploads results to your Foundry project if configured:
    # azure_ai_project={"subscription_id": ..., "resource_group_name": ..., "project_name": ...},
)

print("\nAggregate metrics:")
for name, value in (results.get("metrics") or {}).items():
    print(f"  {name}: {value}")

print("\nPer-row (first columns):")
for row in results.get("rows", [])[:5]:
    print("  " + json.dumps({k: v for k, v in row.items() if "score" in k or k == "inputs.query"}))

os.unlink(dataset_path)
