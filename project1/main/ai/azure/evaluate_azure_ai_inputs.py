from dotenv import load_dotenv
from azure.ai.evaluation import evaluate
from typing import List, Tuple, Dict, Optional, TypedDict
from pathlib import Path
import pandas as pd
import os
from pprint import pprint

load_dotenv(dotenv_path="C:/sandeep/work/trails/pythonpro/project1/azure_env/.env")

from azure.ai.evaluation import evaluate
from azure.ai.evaluation import RelevanceEvaluator

from askwiki import ask_wiki

azure_openai_api_version = os.environ["AZURE_OPENAI_API_VERSION"]
azure_openai_deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
azure_openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
azure_openai_key = os.environ["AZURE_OPENAI_API_KEY"]
path_to_data = os.environ["TEST_DATA_DIR_PATH"]

model_config = {
    "azure_endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT"),
    "azure_deployment": os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
    "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
}

# Ask to wikipedia and evaluate it
def ask_wiki():
    ask_wiki(query="What is the capital of India?")

def run_basic_evaluator():
    relevance_evaluator = RelevanceEvaluator(model_config)
    results = evaluate(
        data=path_to_data + "/data.jsonl",
        target=ask_wiki,
        evaluators={
            "relevance": relevance_evaluator,
        },
    )
    pprint(results)
    pd.DataFrame(results["rows"])

# Multiple Input

# Underlying evaluation: The return ratio of the query to response lengths
def query_response_ratio(query: str, response: str) -> float:
    return len(query) / len(response)

# Helper function that converts a conversation into a list of query-response pairs
def unwrap_conversation(conversation: Dict) -> List[Tuple[str, str]]:
    queries = []
    responses = []
    for turn in conversation["messages"]:
        if turn["role"] == "user":
            queries.append(turn["content"])
        else:
            responses.append(turn["content"])
    return zip(queries, responses)


# Define the output of the evaluation to make the sample repo's robust type requirements happy.
class EvalOutput(TypedDict, total=False):
    result: float


# Actual evaluation function, which handles either a single query-response pair or a conversation
def simple_evaluator_function(
    query: Optional[str] = None, 
    response: Optional[str] = None, 
    conversation: Optional[str] = None
) -> EvalOutput:
    if conversation is not None and query is None and response is None:
        per_turn_results = [query_response_ratio(q, r) for q, r in unwrap_conversation(conversation)]
        return {"result": sum(per_turn_results) / len(per_turn_results), "per_turn_results": per_turn_results}
    if conversation is None and query is not None and response is not None:
        return {"result": query_response_ratio(query, response)}
    raise ValueError("Either a conversation or a query-response pair must be provided.")

# Query/Response Evaluation
def simple_query_and_response_evaluate():
    # Query+response evaluation
    qr_result = simple_evaluator_function(query="What is SBCP?", response="Sopra Banking Cloud Platform (SBCP) is a cloud-based platform for developing applications using microservices architecture.")
    print(f"query/response output: {qr_result}")

# Conversation Evaluation
def simple_conversation_evaluate():
    conversation_input = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "world"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "world and more words to change ratio"},
        ]
    }
    # Conversation evaluation
    conversation_result = simple_evaluator_function(conversation=conversation_input)
    print(f"conversation output: {conversation_result}")

# Query/Response JSONL Evaluation
def jsonl_qr_evaluate():
    # Define data path variables.
    qr_js_data = path_to_data + "/qr_data.jsonl"
    # Change variable referenced here to check different files
    with Path(qr_js_data).open() as f:
        print(f.read())
    # JSONL
    js_qr_output = evaluate(
        data=qr_js_data,
        evaluators={"test": simple_evaluator_function},
        _use_pf_client=False,  # Avoid using PF dependencies to further simplify the example
    )
    eval_row_results = [row["outputs.test.result"] for row in js_qr_output["rows"]]
    metrics = js_qr_output["metrics"]
    print(f"query/response jsonl results: {eval_row_results} \nwith overall metrics: {metrics}")

# Conversation JSONL Evaluation
def jsonl_conversation_evaluate():
    conversation_js_data = path_to_data + "/conversation_data.jsonl"
    js_convo_output = evaluate(
        data=conversation_js_data,
        evaluators={"test": simple_evaluator_function},
        _use_pf_client=False,
    )
    eval_row_results = [row["outputs.test.result"] for row in js_convo_output["rows"]]
    per_turn_results = [row["outputs.test.per_turn_results"] for row in js_convo_output["rows"]]
    metrics = js_convo_output["metrics"]
    print(f"""conversation jsonl results: {eval_row_results} with per turn results: {per_turn_results} and overall metrics: {metrics}""")

# CSV Evaluation
def csv_query_evaluate():
    qr_csv_data = path_to_data + "/qr_data.csv"
    # CSV
    csv_qr_output = evaluate(
        data=qr_csv_data,
        evaluators={"test": simple_evaluator_function},
        _use_pf_client=False,
    )
    eval_row_results = [row["outputs.test.result"] for row in csv_qr_output["rows"]]
    metrics = csv_qr_output["metrics"]
    print(f"Query/response csv results: {eval_row_results} \nwith overall metrics: {metrics}")

#run_basic_evaluator()
simple_query_and_response_evaluate()
#simple_conversation_evaluate()
#jsonl_qr_evaluate()
#jsonl_conversation_evaluate()
#csv_query_evaluate()