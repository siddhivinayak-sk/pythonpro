# pip install --pre "azure-ai-projects>=2.0.0b1" azure-identity python-dotenv
#
# Azure AI Foundry Agents - agent with built-in and custom tools
#
# Creates a persistent agent that has:
#   * File Search (grounded on files you upload to a vector store)
#   * Code Interpreter (runs Python in a sandbox)
#   * A custom Python function tool
#
# The agent then answers a user question, invoking whichever tools it
# decides are needed. Threads let you keep multi-turn state on the server.

import os
import json
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.agents.models import (
    FileSearchTool,
    CodeInterpreterTool,
    FunctionTool,
    ToolSet,
    FilePurpose,
    ListSortOrder,
)
from dotenv import load_dotenv

load_dotenv()

# Full project endpoint, e.g.
#   https://<resource>.services.ai.azure.com/api/projects/<project>
project_endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
model_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")

project_client = AIProjectClient(endpoint=project_endpoint, credential=DefaultAzureCredential())
agents_client = project_client.agents

# --- 1. Custom function tool the agent can call --------------------------

def get_stock_price(symbol: str) -> str:
    """Fake pricing tool. Replace with a real market data call."""
    prices = {"MSFT": 431.20, "GOOG": 178.55, "NVDA": 895.03}
    return json.dumps({"symbol": symbol.upper(), "price_usd": prices.get(symbol.upper(), 100.0)})

user_functions = FunctionTool(functions={get_stock_price})

# --- 2. File Search over an uploaded doc --------------------------------

sample_doc = "sample_earnings.txt"
with open(sample_doc, "w", encoding="utf-8") as f:
    f.write("Q3 revenue was $65.6B, up 22% YoY. Cloud grew 30%.")

uploaded = agents_client.files.upload_and_poll(file_path=sample_doc, purpose=FilePurpose.AGENTS)
vector_store = agents_client.vector_stores.create_and_poll(
    file_ids=[uploaded.id], name="earnings-vs"
)
file_search = FileSearchTool(vector_store_ids=[vector_store.id])

# --- 3. Compose the toolset --------------------------------------------

code_interpreter = CodeInterpreterTool()
toolset = ToolSet()
toolset.add(file_search)
toolset.add(code_interpreter)
toolset.add(user_functions)

# Register function executors so the SDK auto-runs local functions on tool calls
agents_client.enable_auto_function_calls(toolset)

# --- 4. Create the agent, thread, and run ------------------------------

agent = agents_client.create_agent(
    model=model_deployment,
    name="analyst-agent",
    instructions=(
        "You are a financial analyst. Use file_search for company documents, "
        "get_stock_price for live prices, and code_interpreter for math."
    ),
    toolset=toolset,
)
print(f"Created agent {agent.id}")

thread = agents_client.threads.create()
agents_client.messages.create(
    thread_id=thread.id,
    role="user",
    content=(
        "What was the Q3 revenue in the uploaded doc, "
        "and if MSFT stock rose 4% from its current price, what would it be?"
    ),
)

run = agents_client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)
print(f"Run status: {run.status}")

# --- 5. Print the conversation -----------------------------------------

for msg in agents_client.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING):
    text = "".join(part.text.value for part in msg.text_messages) if msg.text_messages else ""
    print(f"\n[{msg.role}] {text}")

# --- 6. Cleanup ---------------------------------------------------------

agents_client.vector_stores.delete(vector_store.id)
agents_client.files.delete(uploaded.id)
agents_client.delete_agent(agent.id)
os.remove(sample_doc)
