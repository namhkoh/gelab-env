"""
VisualWebBench Element Grounding (VWB-EG) evaluation using Qwen2.5-VL guided generation.

Dataset: HongxinLi/VWB-EG (413 samples, bbox in 0-1 normalized format)
Official benchmark: https://visualwebbench.github.io/

Usage:
    python eval/eval_vwb_eg.py \
        --model_path Qwen/Qwen2.5-VL-7B-Instruct \
        --num_gpus 3
"""

import argparse
import json
import os
import re

import torch
import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    GenerationConfig,
)
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize


def get_computer_use_prompt(instruction, screen_width, screen_height):
    system_text = (
        'You are a helpful assistant.\n\n\n# Tools\n\n'
        'You may call one or more functions to assist with the user query.\n\n'
        'You are provided with function signatures within XML tags:\n \n'
        '{"type": "function", "function": {"name": "computer_use", '
        '"description": "Use a mouse and keyboard to interact with a computer screen. '
        f'The screen resolution is {screen_width}x{screen_height}. '
        'Click on the center of UI elements.", '
        '"parameters": {"properties": {"action": {"enum": ["left_click"], "type": "string"}, '
        '"coordinate": {"type": "array"}}, "required": ["action"], "type": "object"}}}\n \n\n'
        'For each function call, return a json object with function name and arguments '
        'within XML tags:\n \n{"name":, "arguments":}\n '
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": instruction},
        ]},
    ]


GUIDED_PREFIX = ' \n{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": ['


def _eval_chunk(gpu_id, model_path, samples, result_queue):
    device = f"cuda:{gpu_id}"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, device_map=device, torch_dtype=torch.bfloat16,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path)
    model.generation_config = GenerationConfig(max_new_tokens=100, do_sample=False, temperature=0.0)

    correct = 0
    total = 0
    wrong_format = 0

    for sample in tqdm(samples, desc=f"VWB-EG:GPU{gpu_id}"):
        img = sample["image"].convert("RGB")
        instruction = sample.get("elem_desc", sample.get("detailed_elem_desc", ""))
        box_01 = sample.get("box", [])
        img_w, img_h = img.size

        if not box_01 or len(box_01) < 4:
            total += 1
            continue

        gt_bbox = box_01

        resized_h, resized_w = smart_resize(
            img.height, img.width,
            factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
            min_pixels=processor.image_processor.min_pixels,
            max_pixels=1048576,
        )
        resized_img = img.resize((resized_w, resized_h))

        messages = get_computer_use_prompt(instruction, resized_w, resized_h)
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text_input = text_input + GUIDED_PREFIX

        inputs = processor(
            text=[text_input], images=[resized_img],
            padding=True, return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(**inputs)

        generated = output_ids[:, inputs.input_ids.shape[1]:]
        response = processor.batch_decode(generated, skip_special_tokens=False)[0]
        if "<|im_end|>" in response:
            response = response[:response.index("<|im_end|>")]

        try:
            nums = re.findall(r'[\d.]+', response.split("]")[0] if "]" in response else response)
            nums = [float(n) for n in nums]
            if len(nums) >= 2:
                px, py = nums[0], nums[1]
            else:
                raise ValueError("No coordinates")

            norm_x = px / resized_w
            norm_y = py / resized_h

            if gt_bbox[0] <= norm_x <= gt_bbox[2] and gt_bbox[1] <= norm_y <= gt_bbox[3]:
                correct += 1
        except Exception:
            wrong_format += 1

        total += 1

    result_queue.put({"correct": correct, "total": total, "wrong_format": wrong_format})


def main():
    parser = argparse.ArgumentParser(description="VWB-EG eval with Qwen2.5-VL guided generation")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--cache_dir", default="/ext_hdd2/nhkoh/.cache/huggingface/datasets")
    parser.add_argument("--output_file", default=None)
    args = parser.parse_args()

    print("Loading VWB-EG dataset...")
    ds = load_dataset("HongxinLi/VWB-EG", split="test",
                       cache_dir=args.cache_dir, trust_remote_code=True)
    all_samples = list(ds)
    print(f"Samples: {len(all_samples)}, GPUs: {args.num_gpus}")

    mp.set_start_method("spawn", force=True)
    q = mp.Queue()

    if args.num_gpus == 1:
        _eval_chunk(0, args.model_path, all_samples, q)
    else:
        chunk_size = (len(all_samples) + args.num_gpus - 1) // args.num_gpus
        processes = []
        for gpu_id in range(args.num_gpus):
            start = gpu_id * chunk_size
            end = min(start + chunk_size, len(all_samples))
            if start >= end:
                continue
            p = mp.Process(target=_eval_chunk, args=(gpu_id, args.model_path, all_samples[start:end], q))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    result = {"correct": 0, "total": 0, "wrong_format": 0}
    while not q.empty():
        r = q.get()
        result["correct"] += r["correct"]
        result["total"] += r["total"]
        result["wrong_format"] += r["wrong_format"]

    accuracy = result["correct"] / result["total"] * 100 if result["total"] > 0 else 0
    result["accuracy"] = accuracy

    print(f"\n{'='*50}")
    print(f"VWB-EG Results")
    print(f"{'='*50}")
    print(f"Accuracy:     {accuracy:.2f}%")
    print(f"Correct:      {result['correct']}/{result['total']}")
    print(f"Wrong format: {result['wrong_format']}")

    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
