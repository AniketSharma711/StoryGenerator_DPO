from huggingface_hub import HfApi

repo_id = "trembl1nghands/Story_Generator_DPO_GPT2"
folder_path = "./final_dpo_model"

api = HfApi()

print(f"Uploading {folder_path} to {repo_id}...")

api.upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    repo_type="model",
)

print("Upload complete!")