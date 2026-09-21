import os
import base64
from openai import OpenAI

# Required environment variables (example names)
FOUNDRY_KEY="..."
ENDPOINT="https://YOUR-RESOURCE-NAME.openai.azure.com/openai/v1/"
MODEL_NAME="your-gpt-image-deployment-name"  # e.g., "gpt-image-1"

client = OpenAI(
    api_key=os.environ["FOUNDRY_KEY"],
    base_url=os.environ["ENDPOINT"],
)

prompt = "A modern flat illustration of a robot holding a potted plant, clean vector style, pastel colors."

response = client.responses.create(
    model=os.environ["MODEL_NAME"],  # your deployment name in Foundry
    input=prompt,
    tools=[{"type": "image_generation"}],
)

image_base64 = next(
    item.result for item in response.output
    if item.type == "image_generation_call"
)

with open("foundry_generated.png", "wb") as f:
    f.write(base64.b64decode(image_base64))

print("Saved: foundry_generated.png")

# Another way
img = client.images.generate(
     model=deployment_name,
     prompt="A cute baby polar bear",
     n=1,
     size="1024x1024",
 )
    
 image_bytes = base64.b64decode(img.data[0].b64_json)
 with open("output.png", "wb") as f:
     f.write(image_bytes)