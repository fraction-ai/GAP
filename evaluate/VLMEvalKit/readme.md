# VLM Eval Kit Quickstart Guide

This guide walks through the quickstart steps for using the VLM Eval Kit to evaluate pre-trained vision-language models. Follow these instructions to set up the environment and begin using the evaluation kit.

## Prerequisites

1. **Create a virtual environment**:

   It's recommended to create and activate a virtual environment for installing dependencies:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # For Linux/macOS
   ```

2. **Install dependencies**:

   Once the virtual environment is activated, install the required dependencies using:

   ```bash
   pip install -e .
   ```

3. **Install `flash-attn`** (required for Qwen2-VL evaluation):

   For optimized attention mechanisms, you can install `flash-attn`.:

   ```bash
   pip install flash-attn
   ```

## Preparing Your Evaluation

You will need to download and set the lora checkpoint before evaluation. You can download the checkpoints using the corresponding script in the `scripts` section.

- MiniCPM-Llama3-V-2_5: Update the checkpoint variable in `vlmeval/vlm/minicpm_v.py`
- Qwen2-VL: Update the checkpoint variable in `vlmeval/vlm/qwen2_vl/model.py`

## Running an Evaluation

You can run the script with python or torchrun for any of the three models - MiniCPM-Llama3-V-2_5, Qwen2-VL-7B-Instruct or Qwen2-VL-2B-Instruct:

```bash
# When running with `python`, only one VLM instance is instantiated, and it might use multiple GPUs (depending on its default behavior).
python run.py --data MME --model MiniCPM-Llama3-V-2_5

# When running with `torchrun`, one VLM instance is instantiated on each GPU. It can speed up the inference.
# However, that is only suitable for VLMs that consume small amounts of GPU memory.

# On a node with 2 GPU
torchrun --nproc-per-node=2 run.py --data MME --model MiniCPM-Llama3-V-2_5
```

---