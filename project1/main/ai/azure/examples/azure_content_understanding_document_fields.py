"""
A sample to demonstrate analyzing with Azure AI Content Understanding Python SDK.

Requirements:
    - Python 3.9 or later

Setup:
    Follow the steps in the URL below to configure your Microsoft Foundry resource and model deployments:
    https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/contentunderstanding/azure-ai-contentunderstanding#configuring-microsoft-foundry-resource

Configuration:
    Before running, update the following variables in the script:
    - AZURE_CONTENT_UNDERSTANDING_ENDPOINT: The endpoint to your Content Understanding resource.
    - CONTENT_UNDERSTANDING_KEY: Your Content Understanding API key (optional if using DefaultAzureCredential).
    - FILE_URL: URL of the file to analyze.

Usage:
    1. Navigate to the directory containing this file:
       # In your terminal
       cd path/to/the/directory/containing/this/file

    2. (Optional) Create and activate a virtual environment:
       python -m venv .venv         # One time setup
       source .venv/bin/activate      # On Linux/macOS
       .venv\\Scripts\\activate        # On Windows

    3. Install dependencies:
       python -m pip install --pre azure-ai-contentunderstanding
       python -m pip install azure-identity

    4. Run the script:
       python sample.py
"""

import json
from urllib.parse import urlparse
from azure.ai.contentunderstanding import ContentUnderstandingClient
from azure.ai.contentunderstanding.models import AnalysisInput, AnalysisResult
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
endpoint = os.getenv("AZURE_COGNITIVE_SERVICE_ENDPOINT", "")
key = os.getenv("AZURE_COGNITIVE_SERVICE_KEY", "")


def is_absolute_url(value: str) -> bool:
    parsed_url = urlparse(value)
    return bool(parsed_url.scheme and parsed_url.netloc)


def main() -> None:
    file_url = "https://github.com/Azure-Samples/azure-ai-content-understanding-python/raw/refs/heads/main/data/invoice.pdf"

    # ANALYZER_ID - the ID of the analyzer to use.
    analyzer_id = "prebuilt-documentFields"

    # API_VERSION - the API version to use.
    api_version = "2026-06-01-preview"

    # Validate Endpoint
    if not is_absolute_url(endpoint):
        print("[Error] Invalid Endpoint. Please provide a valid 'Endpoint'.")
        return

    # Validate File URL
    if not is_absolute_url(file_url):
        print("[Error] Invalid File URL. Please provide a valid URL.")
        return

    # Set up Content Understanding client.
    hasKey = type(key) is str and bool(key.strip()) and "{{CONTENT_UNDERSTANDING_KEY}}" not in key
    credential = AzureKeyCredential(key) if hasKey else DefaultAzureCredential()
    client = ContentUnderstandingClient(endpoint=endpoint, credential=credential, api_version=api_version)

    # [START analyze]
    print(f"Analyzing with {analyzer_id} analyzer...")
    print(f"  File URL: {file_url}\n")

    try:
        poller = client.begin_analyze(
            analyzer_id=analyzer_id,
            inputs=[AnalysisInput(url=file_url)],
        )
        result: AnalysisResult = poller.result()
    except AzureError as err:
        print(f"[Azure Error]: {err.message}")
        return
    except Exception as ex:
        print(f"[Unexpected Error]: {ex}")
        return
    # [END analyze]

    # [START output_result]
    print("=" * 50)
    print("Analysis result:")
    print("=" * 50 + "\n")

    max_display_lines = 50
    result_str = json.dumps(result.as_dict(), indent=2)
    ret_lines = result_str.splitlines()

    if len(ret_lines) > max_display_lines:
        print("\n".join(ret_lines[:max_display_lines]))
        print(f"\n {len(ret_lines) - max_display_lines} more lines to be displayed...\n")
    else:
        print(result_str)

    with open("c:/sandeep/daily/doc-field-output.json", "w", encoding="utf-8") as f:
        f.write(result_str)

    # [END output_result]


if __name__ == "__main__":
    main()