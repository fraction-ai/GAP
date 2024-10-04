from PIL import Image
import requests
from transformers import AutoModel, AutoTokenizer
import torch
from datetime import datetime
import time
from peft import PeftModel
import torch

path_to_adapter="<set_path>"


model_id = "openbmb/MiniCPM-Llama3-V-2_5"
model = AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float16, low_cpu_mem_usage=True)
# model = model.to(device='cuda')
lora_model = PeftModel.from_pretrained(
    model,
    path_to_adapter,
    device_map="auto",
    trust_remote_code=True
).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def generate_response_finetune(image_url, messages):
    try:
        image = Image.open(requests.get(image_url, stream=True).raw)

        res = lora_model.chat(
            image=image,
            msgs=messages,
            tokenizer=tokenizer,
            sampling=True, # if sampling=False, beam_search will be used by default
            temperature=0.7,
            max_tokens=150,
            max_new_tokens=150,
            max_length=150,
            # system_prompt='' # pass system_prompt if needed
        )

        torch.cuda.empty_cache()
        return res
    except Exception as e:
        print(e)
