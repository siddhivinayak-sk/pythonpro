# pip install unstructured
# pip install "unstructured[md]"
# pip install python-magic
# pip install python-magic-bin

import os
from langchain_community.document_loaders import DirectoryLoader
from dotenv import load_dotenv
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from ragas.testset import TestsetGenerator
import pandas as pd
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.graph import Node, NodeType
from ragas.testset.transforms import default_transforms, apply_transforms
from ragas.testset.synthesizers import default_query_distribution



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

llm = LangchainLLMWrapper(AzureChatOpenAI(
    openai_api_version="2023-05-15",
    azure_endpoint=azure_config["base_url"],
    azure_deployment=azure_config["model_deployment"],
    model=azure_config["model_name"],
    validate_base_url=False,
))

embeddings = LangchainEmbeddingsWrapper(AzureOpenAIEmbeddings(
    openai_api_version="2023-05-15",
    azure_endpoint=azure_config["base_url"],
    azure_deployment=azure_config["embedding_deployment"],
    model=azure_config["embedding_name"],
))

def load_dataset():
    loader = DirectoryLoader(path_to_data + "/markdown_dataset", glob="**/*.md")
    docs = loader.load()
    return docs

def simulate_queries():
    doc_dataset = load_dataset()
    generator = TestsetGenerator(llm=llm, embedding_model=embeddings)
    dataset = generator.generate_with_langchain_docs(doc_dataset, testset_size=10)
    df = dataset.to_pandas()
    # Show all columns and rows
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    # Save DataFrame to a CSV file
    output_file = "./output_dir/output.csv"  # Specify the file name
    df.to_csv(output_file, index=False)  # Save without the index column
    print(f"DataFrame content saved to {output_file}")


def create_knowledge_graph():
    kg = KnowledgeGraph()
    data_set = load_dataset()
    for doc in data_set:
        kg.nodes.append(
            Node(
                type=NodeType.DOCUMENT,
                properties={"page_content": doc.page_content, "document_metadata": doc.metadata}
            )
        )
    # define your LLM and Embedding Model
    # here we are using the same LLM and Embedding Model that we used to generate the testset
    transformer_llm = llm
    embedding_model = embeddings

    trans = default_transforms(documents=data_set, llm=transformer_llm, embedding_model=embedding_model)
    apply_transforms(kg, trans)
    kg.save("./output_dir/knowledge_graph.json")
    loaded_kg = KnowledgeGraph.load("./output_dir/knowledge_graph.json")
    return loaded_kg

def simulate_using_testset():
    kg = create_knowledge_graph()
    generator = TestsetGenerator(llm=llm, embedding_model=embeddings)
    query_distribution = default_query_distribution(llm)
    testset = generator.generate(testset_size=10, query_distribution=query_distribution)
    df = testset.to_pandas()
    # Show all columns and rows
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    # Save DataFrame to a CSV file
    output_file = "./output_dir/output_using_kg.csv"  # Specify the file name
    df.to_csv(output_file, index=False)  # Save without the index column
    print(f"DataFrame content saved to {output_file}")



# simulate_queries()
simulate_using_testset()
