from PIL import Image
import requests
from transformers import AutoModel, AutoTokenizer
import torch
import logging

# Configure CloudWatch logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_id = "openbmb/MiniCPM-Llama3-V-2_5"
model = AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float16, low_cpu_mem_usage=True)
model = model.to(device='cuda')
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model.eval()

def generate_response(image_url, messages):
    try:
        image = Image.open(requests.get(image_url, stream=True).raw)

        res = model.chat(
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
        logger.error(f"Error in generate_response: {e}")