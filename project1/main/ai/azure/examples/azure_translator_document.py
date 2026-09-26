import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
endpoint = os.getenv("AZURE_COGNITIVE_SERVICE_ENDPOINT", "")
key = os.getenv("AZURE_COGNITIVE_SERVICE_KEY", "")
SUBSCRIPTION_KEY = key

path = "/translator/document:translate"
url = endpoint + path

headers = {
    "Ocp-Apim-Subscription-Key": SUBSCRIPTION_KEY,
}

# Define the parameters 
# Get list of supported languages and code here: https://aka.ms/TranslatorLanguageCodes 
params = {
    "sourceLanguage": "en-US",
    "targetLanguage": "fr-FR",
    "api-version": "2024-05-01"
}

# Include full path, file name and extension
input_file = "C:/sandeep/daily/20260924/data.txt"
output_file = "C:/sandeep/daily/20260924/data_translated.txt"

# Open the input file in binary mode
with open(input_file, "rb") as document:
    # Define the data to be sent
    # Find list of supported content types here: https://aka.ms/dtsync-content-type
    data = {
        "document": (os.path.basename(input_file), document, "text/plain")
    }

    print(data)

    # Send the POST request
    response = requests.post(url, headers=headers, files=data, params=params)

# Write the response content to a file
with open(output_file, "wb") as output_document:
    output_document.write(response.content)