# pip install azure-ai-vision-imageanalysis python-dotenv
#
# Azure AI Vision - Image Analysis 4.0
# This is a distinct service from GPT vision. It's tuned for structured
# image understanding: captions, tags, objects, people boxes, and OCR.
# Fast, cheap, and returns bounding boxes -- useful for pipelines.

import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from dotenv import load_dotenv

load_dotenv()

endpoint = os.getenv("AZURE_VISION_ENDPOINT")   # https://<name>.cognitiveservices.azure.com/
key = os.getenv("AZURE_VISION_KEY")

client = ImageAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))

image_path = "./static/image/zx-spectrum.jpg"
with open(image_path, "rb") as f:
    image_bytes = f.read()

result = client.analyze(
    image_data=image_bytes,
    visual_features=[
        VisualFeatures.CAPTION,
        VisualFeatures.DENSE_CAPTIONS,
        VisualFeatures.TAGS,
        VisualFeatures.OBJECTS,
        VisualFeatures.PEOPLE,
        VisualFeatures.READ,          # OCR
        VisualFeatures.SMART_CROPS,
    ],
    gender_neutral_caption=True,
    language="en",
)

if result.caption:
    print(f"Caption: '{result.caption.text}'  (confidence={result.caption.confidence:0.2f})")

if result.dense_captions:
    print("\nDense captions:")
    for dc in result.dense_captions.list:
        print(f"  '{dc.text}' @ {dc.bounding_box}")

if result.tags:
    print("\nTags:")
    print("  " + ", ".join(f"{t.name}({t.confidence:0.2f})" for t in result.tags.list))

if result.objects:
    print("\nObjects:")
    for obj in result.objects.list:
        top = obj.tags[0]
        print(f"  {top.name} ({top.confidence:0.2f}) @ {obj.bounding_box}")

if result.people:
    print(f"\nPeople detected: {len(result.people.list)}")

if result.read:
    print("\nOCR:")
    for block in result.read.blocks:
        for line in block.lines:
            print(f"  {line.text}")

if result.smart_crops:
    print("\nSmart crops (by aspect ratio):")
    for c in result.smart_crops.list:
        print(f"  aspect={c.aspect_ratio}  box={c.bounding_box}")
