from huggingface_hub import snapshot_download

# 3 checkpoints - fractionai/MiniCPM-Llama3-V-2.5-checkpoint, fractionai/Qwen2-VL-7B-checkpoint, fractionai/Qwen2-VL-2B-checkpoint
lora_repo = "fractionai/MiniCPM-Llama3-V-2.5-checkpoint"

local_dir = '../evaluate/checkpoints'

# Download the LoRA checkpoint from Hugging Face
lora_checkpoint_path = snapshot_download(repo_id=lora_repo, local_dir=local_dir)
