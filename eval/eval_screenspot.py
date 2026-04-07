"""
ScreenSpot evaluation using Qwen2.5-VL's native computer_use guided generation.

Uses the same prompting strategy as the ScreenSpot-Pro eval framework
(github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding) to reproduce
the paper's base model numbers.

Usage:
    python eval/eval_screenspot.py \
        --model_path Qwen/Qwen2.5-VL-7B-Instruct \
        --gpu 0

    python eval/eval_screenspot.py \
        --model_path checkpoint/gui_exp/sft_amex/v1-.../checkpoint-1537 \
        --gpu 1
"""

import argparse
import json
import os
import re
import sys

import torch
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
)
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize


def get_computer_use_prompt(instruction, screen_width, screen_height):
    """Build Qwen2.5-VL computer_use tool-calling prompt."""
    system_text = f'''You are a helpful assistant.


# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within XML tags:
 
{{"type": "function", "function": {{"name_for_human": "computer_use", "name": "computer_use", "description": "Use a mouse and keyboard to interact with a computer, and take screenshots.\\n* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.\\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn\\'t open, try wait and taking another screenshot.\\n* The screen\\'s resolution is {screen_width}x{screen_height}.\\n* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.\\n* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.\\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don\\'t click boxes on their edges unless asked.", "parameters": {{"properties": {{"action": {{"description": "The action to perform. The available actions are:\\n* `left_click`: Click the left mouse button.", "enum": ["left_click"], "type": "string"}}, "coordinate": {{"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to.", "type": "array"}}}}, "required": ["action"], "type": "object"}}, "args_format": "Format the arguments as a JSON object."}}}}
 

For each function call, return a json object with function name and arguments within XML tags:
 
{{"name":, "arguments":}}
 '''
    return [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": instruction},
        ]},
    ]


GUIDED_PREFIX = ' \n{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": ['


def _eval_chunk(gpu_id, model_path, samples, result_queue):
    """Worker: evaluate a chunk of samples on one GPU."""
    device = f"cuda:{gpu_id}"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, device_map=device, torch_dtype=torch.bfloat16,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path)
    model.generation_config = GenerationConfig(
        max_new_tokens=100, do_sample=False, temperature=0.0,
    )

    correct = 0
    total = 0
    wrong_format = 0

    for i, sample in enumerate(tqdm(samples, desc=f"GPU{gpu_id}")):
        img = sample["image"].convert("RGB")
        instruction = sample["instruction"]
        bbox_01 = sample["bbox"]
        img_w, img_h = img.size

        if bbox_01[2] < 1.5 and bbox_01[3] < 1.5:
            gt_bbox = bbox_01
        else:
            gt_bbox = [bbox_01[0]/img_w, bbox_01[1]/img_h, bbox_01[2]/img_w, bbox_01[3]/img_h]

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
            # The guided prefix already has "coordinate": [
            # Model outputs "X, Y]}}" or "X, Y, X2, Y2]}}"
            # Just parse the numbers from the response
            nums = re.findall(r'[\d.]+', response.split("]")[0] if "]" in response else response)
            nums = [float(n) for n in nums]
            if len(nums) >= 2:
                coords = nums[:2] if len(nums) == 2 else nums[:4]
            else:
                raise ValueError("No coordinates found")
            if len(coords) == 2:
                px, py = coords
            if len(coords) == 2:
                px, py = coords
            elif len(coords) == 4:
                px = (coords[0] + coords[2]) / 2
                py = (coords[1] + coords[3]) / 2
            else:
                raise ValueError("Wrong coord format")

            norm_x = px / resized_w
            norm_y = py / resized_h

            if gt_bbox[0] <= norm_x <= gt_bbox[2] and gt_bbox[1] <= norm_y <= gt_bbox[3]:
                correct += 1
        except Exception:
            wrong_format += 1

        total += 1

    result_queue.put({"correct": correct, "total": total, "wrong_format": wrong_format})


def evaluate_screenspot(model_path, num_gpus, cache_dir, dataset_name="rootsautomation/ScreenSpot"):
    import torch.multiprocessing as mp

    print(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split="test", cache_dir=cache_dir, trust_remote_code=True)
    all_samples = list(ds)
    print(f"Samples: {len(all_samples)}, GPUs: {num_gpus}")

    if num_gpus == 1:
        mp.set_start_method("spawn", force=True)
        q = mp.Queue()
        _eval_chunk(0, model_path, all_samples, q)
        result = q.get()
    else:
        mp.set_start_method("spawn", force=True)
        q = mp.Queue()
        chunk_size = (len(all_samples) + num_gpus - 1) // num_gpus
        processes = []
        for gpu_id in range(num_gpus):
            start = gpu_id * chunk_size
            end = min(start + chunk_size, len(all_samples))
            if start >= end:
                continue
            p = mp.Process(target=_eval_chunk, args=(gpu_id, model_path, all_samples[start:end], q))
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
    print(f"ScreenSpot Results")
    print(f"{'='*50}")
    print(f"Accuracy:     {accuracy:.2f}%")
    print(f"Correct:      {result['correct']}/{result['total']}")
    print(f"Wrong format: {result['wrong_format']}")
    return result


def main():
    parser = argparse.ArgumentParser(description="ScreenSpot eval with Qwen2.5-VL guided generation")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--cache_dir", default="/ext_hdd2/nhkoh/.cache/huggingface/datasets")
    parser.add_argument("--output_file", default=None)
    args = parser.parse_args()

    results = evaluate_screenspot(args.model_path, args.num_gpus, args.cache_dir)

    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
