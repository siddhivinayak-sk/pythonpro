# pip install azure-ai-textanalytics

# Import packages
import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient

# Load environment variables from .env file
load_dotenv()
endpoint = os.getenv("AZURE_COGNITIVE_SERVICE_ENDPOINT", "")
key = os.getenv("AZURE_COGNITIVE_SERVICE_KEY", "")

# Create the client
client = TextAnalyticsClient(endpoint=endpoint, credential=AzureKeyCredential(key))

# Make a request using the client for language detection
text = "¡Hola! Me llamo Josefina y vivo en Madrid, España."
result = client.detect_language([text])[0]

# Print the results
print(f"Language      : {result.primary_language.name}")
print(f"ISO code      : {result.primary_language.iso6391_name}")
print(f"Confidence    : {result.primary_language.confidence_score:.2f}")


# Make request using client for PII detection
text = "Maria Garcia called from 020 7946 0958 and asked to send documents to 42 Market Road, London, UK, SW1A 1AA."
result = client.recognize_pii_entities([text])[0]

# Print the results
print("Redacted text:", result.redacted_text)
print("\nEntities found:")
for entity in result.entities:
    print(f"  {entity.text} | category={entity.category} | confidence={entity.confidence_score}")