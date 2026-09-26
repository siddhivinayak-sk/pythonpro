import requests
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
endpoint = os.getenv("AZURE_COGNITIVE_SERVICE_ENDPOINT", "")
key = os.getenv("AZURE_COGNITIVE_SERVICE_KEY", "")

API_VERSION = "2025-10-01-preview"

SUBSCRIPTION_KEY = key

def translate_text(text, targets, source_language):
    headers = {
        "Ocp-Apim-Subscription-Key": SUBSCRIPTION_KEY,
        "Content-Type": "application/json"
    }
    url = f"{endpoint}/translator/text/translate?api-version={API_VERSION}"
    body = {
        "inputs": [
            {
                "Text": text,
                "language": source_language,
                "targets": targets
            }
        ]
    }

    response = requests.post(url, headers=headers, json=body)
    response.raise_for_status()
    return response.json()
 
def main():
    text = "Doctor is available next Monday. Do you want to schedule an appointment?"
    targets = [
        {"language": "es",},
        {"language": "fr", }
    ]
    source_language = "en"
    try:
        result = translate_text(text, targets,source_language)

        for t in result["value"][0]["translations"]:
            print(f"Translation ({t['language']}): {t['text']}")
    except Exception as e:
        print(f"Translation failed: {e}")
 
if __name__ == "__main__":
    main()