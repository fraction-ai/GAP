from vlmeval.vlm import *
from vlmeval.api import *
from functools import partial


minicpm_series = {
    'MiniCPM-Llama3-V-2_5': partial(MiniCPM_Llama3_V, model_path='openbmb/MiniCPM-Llama3-V-2_5'),
}

qwen2vl_series = {
    'Qwen2-VL-7B-Instruct': partial(Qwen2VLChat, model_path='Qwen/Qwen2-VL-7B-Instruct', min_pixels=1280*28*28, max_pixels=16384*28*28),
    'Qwen2-VL-2B-Instruct': partial(Qwen2VLChat, model_path='Qwen/Qwen2-VL-2B-Instruct', min_pixels=1280*28*28, max_pixels=16384*28*28),
}

supported_VLM = {}

model_groups = [
    minicpm_series, qwen2vl_series
]

for grp in model_groups:
    supported_VLM.update(grp)
