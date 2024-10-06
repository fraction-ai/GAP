# LoRA Fine-Tuning Script

This guide provides instructions for setting up the environment and running LoRA fine-tuning. LoRA enables light-weight model tuning by updating a small subset of parameters while maintaining high efficiency. The script supports LoRA implementation based on the `peft` library.

## Prerequisites

1. Install the required libraries after setting up a Python virtual environment.
   
   Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # For Linux/macOS
   ```

2. Install necessary dependencies:
   ```bash
   pip install torch transformers accelerate deepspeed peft torchvision wandb wheel
   pip install flash_attn
   ```

## Fine-Tuning LoRA

### Command-line Arguments

Download all the images required for finetuning with the given dataset:

```bash
python parallel_download_images.py
```

To launch LoRA fine-tuning, use the following shell script:

```bash
sh finetune_lora.sh
```

This will start the fine-tuning process using the pre-defined configurations and parameters in the script.

### Loading the Fine-Tuned Model

After training is complete, you can load the fine-tuned model with the LoRA adapter using the following Python script:

```python
from peft import PeftModel
from transformers import AutoModel

# Define model type and path to the adapter
model_path = "openbmb/MiniCPM-Llama3-V-2.5"
path_to_adapter = "path_to_your_fine_tuned_checkpoint"

# Load the base pre-trained model
model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True
)

# Load the LoRA adapter and move model to GPU
lora_model = PeftModel.from_pretrained(
    model,
    path_to_adapter,
    device_map="auto",
    trust_remote_code=True
).eval().cuda()
```
