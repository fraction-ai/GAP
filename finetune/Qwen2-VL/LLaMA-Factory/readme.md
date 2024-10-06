# LLaMA Factory Fine-Tuning Script for Qwen2-VL Models

This guide provides instructions for setting up the environment and running LoRA fine-tuning on the `Qwen2-VL-7B` and `Qwen2-VL-2B` models using the [LLaMA Factory repository](https://github.com/hiyouga/LLaMA-Factory). LoRA (Low-Rank Adaptation) allows lightweight model tuning by updating only a small subset of parameters while maintaining efficiency.

## Prerequisites

1. Install the required libraries in a Python virtual environment.

   Create and activate the virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # For Linux/macOS
   ```

2. Install the necessary dependencies for the project:
   ```bash
   pip install -e ".[torch,metrics]"
   ```

This will install all the required libraries, including `torch` and necessary metrics for model finetuning.

## Fine-Tuning Qwen2-VL Models

The fine-tuning process in this repository supports both `Qwen2-VL-7B` and `Qwen2-VL-2B` models. You can use LoRA to fine-tune these models based on your specific dataset and objectives.

### Command-line Arguments

Download all the images required for finetuning with the given dataset:

```bash
cd data
python parallel_download_images.py
```

To initiate the fine-tuning process for the `Qwen2-VL` models, use the following script:

```bash
sh finetune_lora.sh
```

This shell script will start the fine-tuning process based on the configurations you set inside it. Make sure to adjust parameters like batch size, learning rate, and dataset paths according to your needs.


In this example:
- The script will fine-tune the selected `Qwen2-VL` model using LoRA.
- Ensure that the paths and configuration inside the `finetune_qwen2vl.sh` script are correctly set for your project.

### Loading the Fine-Tuned Qwen2-VL Model

After training is complete, you can load the fine-tuned `Qwen2-VL` model with its LoRA adapter using the following Python script:

```python
from peft import PeftModel
from transformers import Qwen2VLForConditionalGeneration

# Define the model type and path to the fine-tuned LoRA adapter
model_path = "Qwen/Qwen2-VL-7B"  # Or "Qwen/Qwen2-VL-2B" for the smaller model
path_to_adapter = "path_to_your_fine_tuned_checkpoint"

# Load the base pre-trained model
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype='auto', device_map='cpu', attn_implementation='flash_attention_2'
)

# Load the LoRA adapter and move the model to GPU
lora_model = PeftModel.from_pretrained(
    model,
    path_to_adapter,
    device_map="auto",  # Automatically allocate model layers to available devices
    trust_remote_code=True
).eval().cuda()
```
