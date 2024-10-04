from PIL import Image
import requests
import torch
import logging
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel

# Configure CloudWatch logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

checkpoint='<checkpoint>'

model_path = "Qwen/Qwen2-VL-7B-Instruct"
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype='auto', device_map='cpu', attn_implementation='flash_attention_2'
)
model = PeftModel.from_pretrained(
    base_model,               # The base Qwen2VL model
    checkpoint,          # Your fine-tuned checkpoint path
    device_map="auto",
    low_cpu_mem_usage=True
).eval().cuda()
processor = AutoProcessor.from_pretrained(model_path)

def generate_response(image_url, messages):
    try:
        # Prepare message
        qwen_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_url
                    },
                    {"type": "text", "text": messages[0]["content"]},
                    {"type": "text", "text": messages[1]["content"]},
                ]
            }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]
    except Exception as e:
        print(e)
        logger.error(f"Error in generate_response: {e}")