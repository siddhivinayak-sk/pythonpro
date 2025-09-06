import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()
path_to_data = os.getenv("TEST_DATA_DIR_PATH")

# api = HfApi(token=os.getenv("HF_TOKEN"))
# api.upload_folder(
#     folder_path= path_to_data + "/ht_dataset",
#     repo_id="siddhivinayak-sk/sk-test-first-dataset",
#     repo_type="dataset",
# )

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path= path_to_data + "/markdown_dataset",
    repo_id="siddhivinayak-sk/sktdataset-gen",
    repo_type="dataset",
)
