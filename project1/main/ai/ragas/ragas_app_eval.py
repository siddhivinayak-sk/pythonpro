# pip install ragas
# pip install sacrebleu
# pip install langchain-openai
# pip install python-dotenv

import os
import asyncio
from dotenv import load_dotenv
from ragas import SingleTurnSample
from ragas.metrics import BleuScore
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import SingleTurnSample
from ragas.metrics import AspectCritic
from datasets import load_dataset
from ragas import EvaluationDataset
from ragas import evaluate

load_dotenv()

path_to_data = os.getenv("TEST_DATA_DIR_PATH")
azure_config = {
    "base_url": os.getenv("AZURE_OPENAI_ENDPOINT"),
    "model_deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    "model_name": os.getenv("AZURE_OPENAI_MODEL"),
    "embedding_deployment": os.getenv("AZURE_OPENAI_EM_DEPLOYMENT"),
    "embedding_name": os.getenv("AZURE_OPENAI_EM_MODEL"),
    "open_ai_version": os.getenv("AZURE_OPENAI_API_VERSION"),
}

evaluator_llm = LangchainLLMWrapper(AzureChatOpenAI(
    openai_api_version="2023-05-15",
    azure_endpoint=azure_config["base_url"],
    azure_deployment=azure_config["model_deployment"],
    model=azure_config["model_name"],
    validate_base_url=False,
))

evaluator_embeddings = LangchainEmbeddingsWrapper(AzureOpenAIEmbeddings(
    openai_api_version="2023-05-15",
    azure_endpoint=azure_config["base_url"],
    azure_deployment=azure_config["embedding_deployment"],
    model=azure_config["embedding_name"],
))

def simple_raga_blue_eval():
    test_data = {
        "user_input": "summarise given text\nThe company reported an 8% rise in Q3 2024, driven by strong performance in the Asian market. Sales in this region have significantly contributed to the overall growth. Analysts attribute this success to strategic marketing and product localization. The positive trend in the Asian market is expected to continue into the next quarter.",
        "response": "The company experienced an 8% increase in Q3 2024, largely due to effective marketing strategies and product adaptation, with expectations of continued growth in the coming quarter.",
        "reference": "The company reported an 8% growth in Q3 2024, primarily driven by strong sales in the Asian market, attributed to strategic marketing and localized products, with continued growth anticipated in the next quarter."
    }
    metric = BleuScore()
    test_data = SingleTurnSample(**test_data)
    value = metric.single_turn_score(test_data)
    print(value)


async def simple_eval_with_llm():
    test_data = {
        "user_input": "summarise given text\nThe company reported an 8% rise in Q3 2024, driven by strong performance in the Asian market. Sales in this region have significantly contributed to the overall growth. Analysts attribute this success to strategic marketing and product localization. The positive trend in the Asian market is expected to continue into the next quarter.",
        "response": "The company experienced an 8% increase in Q3 2024, largely due to effective marketing strategies and product adaptation, with expectations of continued growth in the coming quarter.",
    }

    metric = AspectCritic(name="summary_accuracy",llm=evaluator_llm, definition="Verify if the summary is accurate.")
    test_data = SingleTurnSample(**test_data)
    value = await metric.single_turn_ascore(test_data)
    print(value)

def simple_eval_with_llm_run():
    asyncio.run(simple_eval_with_llm())

def load_dataset_from_file():
    dataset_name = "siddhivinayak-sk/sk-test-first-dataset"
    eval_dataset = load_dataset(dataset_name, split="train")
    eval_dataset = EvaluationDataset.from_hf_dataset(eval_dataset)
    print("Features in dataset:", eval_dataset.features())
    print("Total samples in dataset:", len(eval_dataset))
    return eval_dataset        

def simple_eval_from_dataset():
    eval_dataset = load_dataset_from_file()
    metric = AspectCritic(name="summary_accuracy",llm=evaluator_llm, definition="Verify if the summary is accurate.")
    results = evaluate(eval_dataset, metrics=[metric])
    print("Evaluation Results:", results)
    print(results.to_pandas())


# simple_raga_blue_eval()
# simple_eval_with_llm_run()
simple_eval_from_dataset()