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
from rag_cls import RAG
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness

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

llm = AzureChatOpenAI(
    openai_api_version="2023-05-15",
    azure_endpoint=azure_config["base_url"],
    azure_deployment=azure_config["model_deployment"],
    model=azure_config["model_name"],
    validate_base_url=False,
)

embeddings = AzureOpenAIEmbeddings(
    openai_api_version="2023-05-15",
    azure_endpoint=azure_config["base_url"],
    azure_deployment=azure_config["embedding_deployment"],
    model=azure_config["embedding_name"],
)

evaluator_llm = LangchainLLMWrapper(llm)

sample_docs = [
    "Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity.",
    "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity and won two Nobel Prizes.",
    "Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics.",
    "Charles Darwin introduced the theory of evolution by natural selection in his book 'On the Origin of Species'.",
    "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine."
]

sample_queries = [
    "Who introduced the theory of relativity?",
    "Who was the first computer programmer?",
    "What did Isaac Newton contribute to science?",
    "Who won two Nobel Prizes for research on radioactivity?",
    "What is the theory of evolution by natural selection?"
]

expected_responses = [
    "Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity.",
    "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine.",
    "Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics.",
    "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity and won two Nobel Prizes.",
    "Charles Darwin introduced the theory of evolution by natural selection in his book 'On the Origin of Species'."
]

def simple_rag_test():
    # Initialize RAG instance
    rag = RAG(llm, embeddings)

    # Load documents
    rag.load_documents(sample_docs)

    # Query and retrieve the most relevant document
    query = "Who introduced the theory of relativity?"
    relevant_doc = rag.get_most_relevant_docs(query)

    # Generate an answer
    answer = rag.generate_answer(query, relevant_doc)

    print(f"Query: {query}")
    print(f"Relevant Document: {relevant_doc}")
    print(f"Answer: {answer}")    

def create_data_set():
    dataset = []
    rag = RAG(llm, embeddings)
    rag.load_documents(sample_docs)
    for query,reference in zip(sample_queries,expected_responses):
        relevant_docs = rag.get_most_relevant_docs(query)
        response = rag.generate_answer(query, relevant_docs)
        dataset.append(
            {
                "user_input":query,
                "retrieved_contexts":relevant_docs,
                "response":response,
                "reference":reference
            }
        )
    return EvaluationDataset.from_list(dataset)

def rag_evaluate_from_dataset():
    evaluation_dataset = create_data_set()
    result = evaluate(
        dataset=evaluation_dataset,
        metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
        llm=evaluator_llm
    )
    print(result)

# simple_rag_test()
rag_evaluate_from_dataset()
