"""Manual inference probe for T1.A to see what the model actually emits."""
import os
import sys

os.environ["HF_HOME"] = "/workspace/gelab-env/.cache/huggingface"

import torch
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

CKPT = "/workspace/gelab-env/checkpoint/gui_exp/sft_amex_t1a_aug_full/v0-20260417_130707/checkpoint-4984"

GROUNDING_PROMPT = (
    "I want to {instruction}. "
    "Please locate the target element I should interact with. (with point)"
)
GROUNDING_PROMPT_BASE = (
    "{instruction}\n"
    "Please provide the bounding box coordinate of the region this sentence describes."
)

SYSTEM_PROMPT = (
    "You are a Multifaceted Mobile Interface Assistant. Your responsibilities include:\n"
    "- 1. Navigating a mobile phone interface to reach a target page based on user "
    "instructions, task history, and the current screen state.\n"
    "- 2. Understanding icons by identifying their name or function based on their "
    "location on the screen.\n"
    "- 3. Grounding icons by locating the coordinates of an icon based on its name "
    "or description.\n"
    "For actions involving coordinates, use the (0,0) to (1000,1000) system."
)

print("Loading model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    CKPT,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(CKPT)
model.eval()

print("Loading 5 ScreenSpot samples...")
ds = load_dataset(
    "rootsautomation/ScreenSpot",
    split="test",
    cache_dir="/workspace/gelab-env/.cache/huggingface/datasets",
    trust_remote_code=True,
)

def run(sample, prompt_template, use_system=False):
    img = sample["image"]
    instr = sample["instruction"]
    query = prompt_template.format(instruction=instr)
    messages = []
    if use_system:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": query},
        ],
    })
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    generated = output_ids[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0]

for i in range(5):
    s = ds[i]
    print(f"\n=== sample {i}  |  instr='{s['instruction'][:60]}'  |  bbox={s['bbox']} ===")
    for label, tmpl, use_sys in [
        ("GE-Lab grounding (no sys)", GROUNDING_PROMPT, False),
        ("GE-Lab grounding (+ sys)",  GROUNDING_PROMPT, True),
        ("Qwen-native bbox2d (no sys)", GROUNDING_PROMPT_BASE, False),
    ]:
        resp = run(s, tmpl, use_sys)
        print(f"  [{label}]  ->  {resp[:180]!r}")

print("\nDone.")
