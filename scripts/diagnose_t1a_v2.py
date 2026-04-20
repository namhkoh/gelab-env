"""T1.A diagnostic v2 — show raw tokens, explicit max_pixels, try training-style prompt."""
import os, sys

os.environ["HF_HOME"] = "/workspace/gelab-env/.cache/huggingface"

import torch
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

CKPT = "/workspace/gelab-env/checkpoint/gui_exp/sft_amex_t1a_aug_full/v0-20260417_130707/checkpoint-4984"
MAX_PIXELS = 2592000

print("Loading model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    CKPT, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(CKPT, max_pixels=MAX_PIXELS)
model.eval()

SYSTEM_PROMPT = open("/workspace/gelab-env/scripts/_system_prompt_marker.txt", "r").read().strip() if os.path.exists("/workspace/gelab-env/scripts/_system_prompt_marker.txt") else None

# Load one ScreenSpot sample + one training sample clone
ds = load_dataset("rootsautomation/ScreenSpot", split="test",
                  cache_dir="/workspace/gelab-env/.cache/huggingface/datasets", trust_remote_code=True)

def infer(messages, label):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    gen = output_ids[:, inputs.input_ids.shape[1]:]
    skip = processor.batch_decode(gen, skip_special_tokens=True)[0]
    raw = processor.batch_decode(gen, skip_special_tokens=False)[0]
    print(f"\n[{label}]")
    print(f"  text-in-template (last 300):  {text[-300:]!r}")
    print(f"  n_tokens_in = {inputs.input_ids.shape[1]}  n_gen = {gen.shape[1]}")
    print(f"  SKIP  :  {skip!r}")
    print(f"  RAW   :  {raw!r}")
    return raw

s = ds[0]  # instr='close', bbox far right
print(f"Sample: instr='{s['instruction']}'  bbox={s['bbox']}")

# Test 1: exactly as the training format (single user turn, <image> embedded via image_path string)
msg1 = [{
    "role": "user",
    "content": [
        {"type": "image", "image": s["image"]},
        {"type": "text", "text": f"I want to click on {s['instruction']}. Please locate the target element I should interact with. (with point)"},
    ],
}]
infer(msg1, "default prompt (no sys)")

# Test 2: same as in training literally — <image>-prefixed single string
msg2 = [{
    "role": "user",
    "content": [
        {"type": "image", "image": s["image"]},
        {"type": "text", "text": f"<image>I want to click on {s['instruction']}. Please locate the target element I should interact with. (with point)"},
    ],
}]
infer(msg2, "with explicit <image> in text")

# Test 3: no grounding prompt, just a navigation-style instruction
msg3 = [{
    "role": "user",
    "content": [
        {"type": "image", "image": s["image"]},
        {"type": "text", "text": "Close the window."},
    ],
}]
infer(msg3, "nav-style instruction")

# Test 4: understanding-style query about a specific point
msg4 = [{
    "role": "user",
    "content": [
        {"type": "image", "image": s["image"]},
        {"type": "text", "text": "What is the icon at point (950, 175) in the image?"},
    ],
}]
infer(msg4, "understanding-style query")

print("\nDone.")
