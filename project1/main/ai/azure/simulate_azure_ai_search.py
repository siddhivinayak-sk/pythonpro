import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from openai import AzureOpenAI
import requests
from azure.ai.evaluation.simulator import Simulator
import asyncio
from dotenv import load_dotenv

load_dotenv(dotenv_path="C:/sandeep/work/trails/pythonpro/project1/azure_env/.env")



azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
subscription_key = os.environ["AZURE_OPENAI_API_KEY"]
azure_deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
should_cleanup: bool = False

search_endpoint = os.environ["AZURE_AI_SEARCH_ENDPOINT"]
index_name = os.environ["AZURE_AI_SEARCH_INDEX"]
search_api_key = os.environ["AZURE_AI_SEARCH_KEY"]

model_config = {
    "azure_endpoint": azure_endpoint,
    "azure_deployment": azure_deployment,
    "api_key": subscription_key,
}

simulator = Simulator(model_config=model_config)

def call_to_your_ai_application(query: str) -> str:
    # logic to call your application
    # use a try except block to catch any errors

    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
        api_key=subscription_key,
    )
    completion = client.chat.completions.create(
        model=deployment,
        messages=[
            {
                "role": "user",
                "content": query,
            }
        ],
        max_tokens=800,
        temperature=0.7,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False,
    )
    message = completion.to_dict()["choices"][0]["message"]
    # change this to return the response from your application
    return message["content"]

def generate_text_from_index(search_term: str) -> str:
    url = f"{search_endpoint}/indexes/{index_name}/docs/search?api-version=2024-07-01"
    headers = {
        "Content-Type": "application/json",
        "api-key": search_api_key,
    }
    search_query = {
        "search": search_term, 
        "top": 50,
        "queryType": "semantic",
        "semanticConfiguration": "default"
     }
    response = requests.post(url=url, headers=headers, data=json.dumps(search_query))

    text = ""
    if response.status_code == 200:
        results = response.json()
        for result in results["value"]:
            text += result["content"]

    return text[:5000]

async def callback(
    messages: List[Dict],
    stream: bool = False,
    session_state: Any = None,  # noqa: ANN401
    context: Optional[Dict[str, Any]] = None,
) -> dict:
    messages_list = messages["messages"]
    # get last message
    latest_message = messages_list[-1]
    query = latest_message["content"]
    context = None
    # call your endpoint or ai application here
    response = call_to_your_ai_application(query)
    # we are formatting the response to follow the openAI chat protocol format
    formatted_response = {
        "content": response,
        "role": "assistant",
        "context": {
            "citations": None,
        },
    }
    messages["messages"].append(formatted_response)
    return {"messages": messages["messages"], "stream": stream, "session_state": session_state, "context": context}

async def main():
    term_used_to_search = "Sopra Banking Cloud Platform (SBCP)"
    text = generate_text_from_index(term_used_to_search)

    outputs = await simulator(
        target=callback,
        text=text,
        num_queries=1,
        max_conversation_turns=1,
        tasks=[
            f"I am a technical architect, I want to use {term_used_to_search} for application development, what are purpose for SBCP",
            f"I am a technical architect, I want to use {term_used_to_search} for application development, what are features in SBCP",
            f"I am a developer, I want to use {term_used_to_search} for application development, how to create a microservice using SBCP",
            f"I am a developer, I want to use {term_used_to_search} for application development, how to create an endpoint for REST resource in SBCP",
        ],
    )
    print(outputs)
    #output_file = Path("output.json")
    #with output_file.open("a") as f:
    #    json.dump(outputs, f)

if __name__ == "__main__":
    asyncio.run(main())