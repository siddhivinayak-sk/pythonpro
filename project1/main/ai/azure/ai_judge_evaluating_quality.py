# Installation packages
# pip install azure-ai-evaluation
# pip install promptflow-azure
# pip install azure-identity
# pip install --upgrade openai
# pip install python-dotenv

from dotenv import load_dotenv
from pprint import pprint
import pandas as pd
from azure.identity import DefaultAzureCredential
import os
import pathlib
from azure.ai.evaluation import RougeType
from azure.ai.evaluation import evaluate
from azure.ai.evaluation import (
    ContentSafetyEvaluator,
    RelevanceEvaluator,
    CoherenceEvaluator,
    GroundednessEvaluator,
    FluencyEvaluator,
    SimilarityEvaluator,
    BleuScoreEvaluator,
    F1ScoreEvaluator,
    GleuScoreEvaluator,
    IntentResolutionEvaluator,
    MeteorScoreEvaluator,
    ResponseCompletenessEvaluator,
    RetrievalEvaluator,
    RougeScoreEvaluator,
    SimilarityEvaluator,
    TaskAdherenceEvaluator,
    ToolCallAccuracyEvaluator,
)
from query_model_endpoint import QueryModelEndpoint

load_dotenv(dotenv_path="C:/sandeep/work/trails/pythonpro/project1/azure_env/.env")

azure_ai_project = {
    "subscription_id": os.getenv("AZURE_SUBSCRIPTION_ID"),
    "resource_group_name": os.getenv("AZURE_RESOURCE_GROUP"),
    "project_name": os.getenv("AZURE_PROJECT_NAME"),
}

azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_key = os.getenv("AZURE_OPENAI_API_KEY")
path_to_data = os.getenv("TEST_DATA_DIR_PATH")
search_endpoint = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
search_key = os.getenv("AZURE_AI_SEARCH_KEY")
search_index = os.getenv("AZURE_AI_SEARCH_INDEX")

model_config = {
    "azure_endpoint": azure_openai_endpoint,
    "azure_deployment": azure_openai_deployment,
    "api_key": azure_openai_key,
}

search_config = {
    "search_endpoint": search_endpoint,
    "search_key": search_key,
    "search_index": search_index
}

def load_dataset():
    df = pd.read_json(path_to_data + "/test_data.jsonl", lines=True)
    print(df.head())
    return df

def run_evaluators():
    #content_safety_evaluator = ContentSafetyEvaluator(azure_ai_project=azure_ai_project, credential=DefaultAzureCredential())
    relevance_evaluator = RelevanceEvaluator(model_config)
    coherence_evaluator = CoherenceEvaluator(model_config)
    groundedness_evaluator = GroundednessEvaluator(model_config)
    fluency_evaluator = FluencyEvaluator(model_config)
    similarity_evaluator = SimilarityEvaluator(model_config)
    bleu_score_evaluator = BleuScoreEvaluator()
    f1_scope_evaluator = F1ScoreEvaluator()
    gleu_score_evaluator = GleuScoreEvaluator()
    intent_resolution_evaluator = IntentResolutionEvaluator(model_config)
    meteor_score_evaluator = MeteorScoreEvaluator()
    response_completeness_evaluator = ResponseCompletenessEvaluator(model_config)
    retrieval_evaluator = RetrievalEvaluator(model_config)
    rouge_score_evaluator = RougeScoreEvaluator(RougeType.ROUGE_4)
    similarity_evaluator = SimilarityEvaluator(model_config)
    task_adherence_evaluator = TaskAdherenceEvaluator(model_config)
    tool_call_accuracy_evaluator = ToolCallAccuracyEvaluator(model_config)


    path = path_to_data + "/test_data.jsonl"
    evaluators = {
            #"content_safety": content_safety_evaluator,
            "coherence": coherence_evaluator,
            "relevance": relevance_evaluator,
            "groundedness": groundedness_evaluator,
            "fluency": fluency_evaluator,
            "similarity": similarity_evaluator,
            "bleu_score": bleu_score_evaluator,
            "f1_score": f1_scope_evaluator,
            "gleu_score": gleu_score_evaluator,
            "intent_resolution": intent_resolution_evaluator,
            "meteor_score": meteor_score_evaluator,
            "response_completeness": response_completeness_evaluator,
            "retrieval": retrieval_evaluator,
            "rouge_score": rouge_score_evaluator,
            "similarity": similarity_evaluator,
            "task_adherence": task_adherence_evaluator,
        }
    evaluator_config = {
            #"content_safety": {"column_mapping": {"query": "${data.query}", "response": "${target.response}"}},
            "coherence": {
                "column_mapping": {"response": "${target.response}", "query": "${data.query}"}
            },
            "relevance": {
                "column_mapping": {"response": "${target.response}", "context": "${data.context}", "query": "${data.query}"}
            },
            "groundedness": {
                "column_mapping": {"response": "${target.response}", "context": "${data.context}", "query": "${data.query}"}
            },
            "fluency": {
                "column_mapping": {"response": "${target.response}", "context": "${data.context}", "query": "${data.query}"}
            },
            "similarity": {
                "column_mapping": {
                    "response": "${target.response}", 
                    "context": "${data.context}", 
                    "query": "${data.query}",
                    "ground_truth": "${data.ground_truth}",
                    #"conversation": "${data.conversation}",
                    }
            },
            "bleu_score": {
                "column_mapping": {"response": "${target.response}", "ground_truth": "${data.ground_truth}"}
            },
            "f1_score": {
                "column_mapping": {"response": "${target.response}", "ground_truth": "${data.ground_truth}"}
            },
            "gleu_score": {
                "column_mapping": {"response": "${target.response}", "ground_truth": "${data.ground_truth}"}
            },
            "intent_resolution": {
                "column_mapping": {"response": "${target.response}", "context": "${data.context}", "query": "${data.query}"}
            },
            "meteor_score": {
                "column_mapping": {"response": "${target.response}", "ground_truth": "${data.ground_truth}"}
            },
            "response_completeness": {
                "column_mapping": {"response": "${target.response}", "ground_truth": "${data.ground_truth}"}
            },
            "retrieval": {
                "column_mapping": {"messages": "${target.response}", "context": "${data.context}"}
            },
            "rouge_score": {
                "column_mapping": {"response": "${target.response}", "ground_truth": "${data.ground_truth}"}
            },
            "similarity": {
                "column_mapping": {"response": "${target.response}", "query": "${data.query}", "ground_truth": "${data.ground_truth}"}
            },
            "task_adherence": {
                "column_mapping": {"response": "${target.response}", "query": "${data.query}", "ground_truth": "${data.ground_truth}"}
            },
            "tool_call_accuracy": {
                "column_mapping": {"response": "${target.response}", "query": "${data.query}", "ground_truth": "${data.ground_truth}"}
            },
        }

    results = evaluate(
        evaluation_name="Eval-Run-" + "-" + model_config["azure_deployment"].title(),
        data=path,
        target=QueryModelEndpoint(model_config, search_config),
        evaluators=evaluators,
        evaluator_config=evaluator_config,
    )
    pprint(results)
    #pd.DataFrame(results["rows"])

run_evaluators()