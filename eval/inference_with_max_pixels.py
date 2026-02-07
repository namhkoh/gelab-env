#!/usr/bin/env python3
"""
Inference script with MAX_PIXELS matching training (200704)
"""

import os
import json
import argparse
from tqdm import tqdm

# Set MAX_PIXELS BEFORE importing vllm/transformers
os.environ['MAX_PIXELS'] = '200704'

from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

# System prompt from training
SYSTEM_PROMPT = """You are a **Multifaceted Mobile Interface Assistant**. Your responsibilities include:
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

**2. Task: Icon Grounding (Locating an Icon)**
   *   **Goal:** Identify the coordinates of a requested icon.
   *   **Typical Input:** User request like "Click on [icon name/description] in the image.", screen image (`<image>`).
   *   **Action:** Implicitly `click` (meaning "identify location").
   *   **Output Format:**
      ```
      Action: click(start_box='<|box_start|>(x,y)<|box_end|>')
      ```
      *(Note: The explanation is often implicit in the grounding request itself).*

**3. Task: Icon Understanding (Identifying an Icon)**
   *   **Goal:** Provide the name or function of an icon at given coordinates.
   *   **Typical Input:** User request like "What is the icon at point (x, y) in the image?", screen image (`<image>`).
   *   **Action:** Provide textual information.
   *   **Output Format:**
      ```
      [Icon Name or Description]
      ```
      *(Just the direct answer as text).*

--- General Instructions ---

*   Carefully analyze the user request to determine the task (Navigation, Grounding, Understanding).
*   Analyze the current screen state (description or image) thoroughly.
*   For actions involving coordinates (`click`), use the (0,0) to (1000,1000) system.
*   Strictly adhere to the specified output format for the determined task type. Use a tab character (`\t`) as a separator where indicated.
*   Before Navigation Task, you need to think about the next action and explain it. The reasoning should explain *why* the provided 'Next Action' is the logical choice at this point. Focus only on generating the text that goes *inside* the <think></think> tags.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--save_file', type=str, required=True)
    args = parser.parse_args()

    print(f"MAX_PIXELS: {os.environ.get('MAX_PIXELS')}")
    print(f"Model: {args.model_path}")
    print(f"Test: {args.test_file}")
    
    # Load processor and set max_pixels to match training
    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.image_processor.max_pixels = 200704
    print(f"Processor max_pixels set to: {processor.image_processor.max_pixels}")
    
    # Load model
    print("Loading model with vLLM...")
    llm = LLM(
        model=args.model_path,
        limit_mm_per_prompt={"image": 10, "video": 10},
        gpu_memory_utilization=0.8,
    )
    
    # Load test data
    with open(args.test_file, 'r') as f:
        test_data = json.load(f)
    
    print(f"Total samples: {len(test_data)}")
    
    sampling_params = SamplingParams(temperature=0, top_p=0.001, max_tokens=200)
    
    results = []
    for sample in tqdm(test_data, desc="Evaluating"):
        # Build message
        content = [
            {"type": "image", "image": sample['images'][0]},
            {"type": "text", "text": sample['messages'][0]['content']}
        ]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ]
        
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        
        outputs = llm.generate(
            [{"prompt": prompt, "multi_modal_data": {"image": image_inputs}}],
            sampling_params=sampling_params
        )
        
        prediction = outputs[0].outputs[0].text
        
        # Preserve all fields from input + add prediction
        result = {
            "idx": sample.get('idx'),
            "path": sample.get('path'),
            "source": sample.get('source'),
            "task": sample.get('task'),
            "images": sample['images'],
            "messages": sample['messages'],
            "bbox_norm": sample.get('bbox_norm'),
            "prediction": prediction
        }
        results.append(result)
        
        # Save incrementally
        with open(args.save_file, 'w') as f:
            for r in results:
                f.write(json.dumps(r) + '\n')
    
    print(f"\nSaved {len(results)} results to {args.save_file}")


if __name__ == "__main__":
    main()
