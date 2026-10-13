# pip install --pre "azure-ai-projects>=2.0.0b1" azure-identity python-dotenv
#
# Azure AI Foundry Agents - minimal example
# Retrieves an existing agent from a Foundry project and asks it a question.

import os
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from dotenv import load_dotenv

load_dotenv()

# Full project endpoint, e.g.
#   https://<resource>.services.ai.azure.com/api/projects/<project>
project_endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
agent_name = os.getenv("AZURE_AI_AGENT_NAME", "learning-agent")

project_client = AIProjectClient(endpoint=project_endpoint, credential=DefaultAzureCredential())

# Get an existing agent
agent = project_client.agents.get(agent_name=agent_name)
print(f"Retrieved agent: {agent.name}")

openai_client = project_client.get_openai_client()

# Reference the agent to get a response
response = openai_client.responses.create(
    input=[{"role": "user", "content": "Tell me what you can help with."}],
    extra_body={"agent": {"name": agent.name, "type": "agent_reference"}},
)

print(f"Response output: {response.output_text}")
