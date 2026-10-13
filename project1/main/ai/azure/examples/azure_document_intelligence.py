# pip install azure-ai-documentintelligence python-dotenv
#
# Azure AI Document Intelligence (formerly Form Recognizer)
# Extracts structured data from documents using prebuilt models:
#   * prebuilt-layout   - text, tables, selection marks, structure
#   * prebuilt-invoice  - invoice fields (vendor, total, line items)
#   * prebuilt-read     - text-only OCR
#
# Content Understanding (elsewhere in this folder) is the newer,
# schema-driven service; Document Intelligence is the mature, prebuilt-
# model service still widely used in production pipelines.

import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from dotenv import load_dotenv

load_dotenv()

endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))

# Public sample invoice from the docs
invoice_url = "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-REST-api-samples/master/curl/form-recognizer/sample-invoice.pdf"

print("Running prebuilt-invoice ...")
poller = client.begin_analyze_document(
    model_id="prebuilt-invoice",
    body=AnalyzeDocumentRequest(url_source=invoice_url),
)
result = poller.result()

for i, doc in enumerate(result.documents or []):
    print(f"\nInvoice #{i + 1} (confidence={doc.confidence:0.2f})")
    for field_name in ("VendorName", "CustomerName", "InvoiceId", "InvoiceDate",
                       "InvoiceTotal", "SubTotal", "TotalTax"):
        field = doc.fields.get(field_name)
        if field:
            print(f"  {field_name}: {field.content}  (conf={field.confidence:0.2f})")

    items = doc.fields.get("Items")
    if items and items.value_array:
        print("  Line items:")
        for line in items.value_array:
            props = line.value_object or {}
            desc = props.get("Description")
            amt = props.get("Amount")
            print(f"    - {desc.content if desc else '?'} => {amt.content if amt else '?'}")

print(f"\nPages: {len(result.pages or [])}, Tables: {len(result.tables or [])}")

# --- Optional: run the prebuilt-layout model on the same doc ------------
# poller = client.begin_analyze_document(
#     model_id="prebuilt-layout",
#     body=AnalyzeDocumentRequest(url_source=invoice_url),
# )
# layout = poller.result()
# for table in layout.tables or []:
#     print(f"Table {table.row_count}x{table.column_count}")
