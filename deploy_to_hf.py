"""
Deploys the Gradio app to HF Spaces.
Merged model loads from HF Hub directly — no model files needed in Space.
"""
import os
import shutil
from huggingface_hub import HfApi, create_repo

HF_USERNAME = input("Enter your Hugging Face username: ").strip()
SPACE_NAME = "hinglish-toxicity-detector"
REPO_ID = f"{HF_USERNAME}/{SPACE_NAME}"

api = HfApi()

print(f"\n1. Creating Space: {REPO_ID}")
create_repo(REPO_ID, repo_type="space", space_sdk="gradio", exist_ok=True)
print(f"   Space: https://huggingface.co/spaces/{REPO_ID}")

deploy_dir = os.path.join(os.path.dirname(__file__), "deploy")

model_dir = os.path.join(deploy_dir, "model")
if os.path.exists(model_dir):
    shutil.rmtree(model_dir)
    print("\n2. Cleaned old model files from deploy folder")

print(f"\n3. Uploading app to HF Space...")
api.upload_folder(
    folder_path=deploy_dir,
    repo_id=REPO_ID,
    repo_type="space",
)

print(f"\n   Done! Live at: https://huggingface.co/spaces/{REPO_ID}")
print(f"   Build takes ~5-10 min. Model (~2.3GB) downloads on first start.")
