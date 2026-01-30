"""
Inference script using HuggingFace transformers (no vllm dependency)
"""

import json
import os
import argparse
from tqdm import tqdm
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

SYSTEM_PROMPT = '''You are a **Multifaceted Mobile Interface Assistant**. Your responsibilities include:
1.  **Navigating** a mobile phone interface to reach a target page based on user instructions, task history, and the current screen state.
2.  **Understanding icons** by identifying their name or function based on their location on the screen.
3.  **Grounding icons** by locating the coordinates of an icon based on its name or description.

You will receive input that typically includes:
*   A **User Request:** Specifies the goal (navigation, understanding, or grounding). This might be a complex instruction for navigation or a direct question/command for icon tasks.
*   **Task History (Optional, primarily for Navigation):** Records previous steps.
*   **Current Screen State:** Represents the current screen, an image (indicated by `<image>`).

**Based on the user request and the current screen state (and history if applicable), you must first determine the type of task requested and then provide the appropriate output.**

--- Task Types and Output Formats ---

**1. Task: Navigation**
   *   **Goal:** Reach a target page step-by-step.
   *   **Typical Input:** Multi-turn instruction, history, and state. screen description and screenshot.
   *   **Possible Actions:**
      *   `click`: Tap a specific element. Provide coordinates (x, y) relative to a (0,0) top-left and (1000,1000) bottom-right system.
      *   `complete`: Task finished, current screen is the target.
   *   **Output Format:**
      ```
      Explain: [Your brief explanation, e.g., 'click xxx icon on yyy page.', 'this is the target page.']\tAction: [click(start_box='<|box_start|>(x,y)<|box_end|>') or complete]  # Include point only for CLICK
      ```

--- General Instructions ---

*   Carefully analyze the user request to determine the task (Navigation, Grounding, Understanding).
*   Analyze the current screen state (description or image) thoroughly.
*   For actions involving coordinates (`click`), use the (0,0) to (1000,1000) system.
*   Strictly adhere to the specified output format for the determined task type. Use a tab character (`\t`) as a separator where indicated.
'''


def load_model(model_path: str, device: str = "cuda"):
    """Load model and processor."""
    print(f"Loading model from {model_path}...")
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    
    return model, processor


def run_inference(model, processor, data: dict, max_new_tokens: int = 512):
    """Run inference on a single sample."""
    
    # Build messages
    content = []
    
    # Add image if present
    if 'images' in data and data['images']:
        image_path = data['images'][0]
        content.append({
            "type": "image",
            "image": image_path
        })
    
    # Add text content
    user_text = data['messages'][0]['content']
    content.append({
        "type": "text",
        "text": user_text
    })
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content}
    ]
    
    # Process inputs
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
    inputs = inputs.to(model.device)
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    
    # Decode
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]


def main():
    parser = argparse.ArgumentParser(description='Run inference with HuggingFace')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--test_file', type=str, required=True, help='Path to test JSON file')
    parser.add_argument('--save_file', type=str, required=True, help='Path to save results')
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples to process')
    parser.add_argument('--max_new_tokens', type=int, default=512, help='Max new tokens to generate')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.save_file) or '.', exist_ok=True)
    
    # Load model
    model, processor = load_model(args.model_path)
    
    # Load test data
    with open(args.test_file, 'r') as f:
        test_data = json.load(f)
    
    if args.max_samples:
        test_data = test_data[:args.max_samples]
    
    print(f"Running inference on {len(test_data)} samples...")
    
    # Check for existing results to resume
    processed_ids = set()
    if os.path.exists(args.save_file):
        with open(args.save_file, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    if 'idx' in record:
                        processed_ids.add(record['idx'])
                except:
                    pass
        print(f"Found {len(processed_ids)} already processed samples, resuming...")
    
    # Run inference
    with open(args.save_file, 'a') as f_out:
        for data in tqdm(test_data, desc="Inference"):
            idx = data.get('idx', -1)
            
            if idx in processed_ids:
                continue
            
            try:
                prediction = run_inference(model, processor, data, args.max_new_tokens)
                data['prediction'] = prediction
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                f_out.flush()
            except Exception as e:
                print(f"\nError on idx {idx}: {e}")
                continue
    
    print(f"\nResults saved to {args.save_file}")


if __name__ == "__main__":
    main()
