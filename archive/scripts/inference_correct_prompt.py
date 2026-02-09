"""
Inference script using the CORRECT system prompt from training.
This fixes the evaluation mismatch issue.
"""

import json
from tqdm import tqdm
import re
import os
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
import argparse

# Load the training system prompt
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(SCRIPT_DIR, 'training_system_prompt.txt'), 'r') as f:
    SYSTEM_PROMPT = f.read()

print(f"Using training system prompt ({len(SYSTEM_PROMPT)} chars)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate with correct system prompt.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--test_file', type=str, required=True, help='Path to test JSON')
    parser.add_argument('--save_file', type=str, required=True, help='Output JSONL path')
    args = parser.parse_args()

    model_path = args.model_path
    eval_file = args.test_file
    save_file = args.save_file

    # Check for resume
    processed_ids = set()
    resuming = False
    if os.path.exists(save_file):
        print(f"Output file {save_file} exists. Reading processed IDs to resume...")
        resuming = True
        try:
            with open(save_file, "r", encoding='utf-8') as f_existing:
                for line in f_existing:
                    try:
                        record = json.loads(line.strip())
                        if 'idx' in record:
                            processed_ids.add(record['idx'])
                    except json.JSONDecodeError:
                        pass
            print(f"Found {len(processed_ids)} already processed IDs.")
        except Exception as e:
            print(f"Error reading existing save file: {e}. Starting from scratch.")
            processed_ids = set()
            resuming = False

    # Initialize vLLM
    llm = LLM(
        model=model_path,
        limit_mm_per_prompt={"image": 10, "video": 10},
    )

    sampling_params = SamplingParams(
        temperature=0,
        top_p=0.001,
        repetition_penalty=1.05,
        max_tokens=1024,
        stop_token_ids=[],
    )

    processor = AutoProcessor.from_pretrained(model_path)

    # Load test data
    with open(eval_file, "r", encoding='utf-8') as f:
        datas = json.load(f)

    open_mode = 'a' if resuming else 'w'
    print(f"Starting processing. Results will be saved to {save_file} (mode: {open_mode})")

    items_processed = 0
    items_skipped = 0

    with open(save_file, open_mode, encoding='utf-8') as f_out:
        for i, data in enumerate(tqdm(datas, desc="Processing data")):
            if 'idx' not in data:
                print(f"Warning: Input item at index {i} is missing 'idx'. Skipping.")
                continue

            idx = data['idx']
            if idx in processed_ids:
                items_skipped += 1
                continue

            # Build message content
            content = []
            if 'images' in data and data['images']:
                content.append({
                    "type": "image",
                    "image": data['images'][0]
                })

            content.append({
                "type": "text",
                "text": data['messages'][0]['content']
            })

            # Use the CORRECT training system prompt
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content}
            ]

            try:
                # vLLM inference
                prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                mm_data = {}
                if image_inputs is not None:
                    mm_data["image"] = image_inputs
                if video_inputs is not None:
                    mm_data["video"] = video_inputs
                llm_inputs = {"prompt": prompt, "multi_modal_data": mm_data}

                outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
                response = outputs[0].outputs[0].text

                data['prediction'] = response

                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                f_out.flush()
                items_processed += 1

            except Exception as e:
                print(f"\nError processing idx {idx}: {e}")
                continue

    print("\n--- Processing Summary ---")
    print(f"Total items in input file: {len(datas)}")
    print(f"Items skipped (already processed): {items_skipped}")
    print(f"Items processed in this run: {items_processed}")
    print("Processing complete.")
