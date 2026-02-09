

import json
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import re
import math
import os
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import traceback # For more detailed error printing if needed

system= '''
You are a **Multifaceted Mobile Interface Assistant**. Your responsibilities include:
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
'''

def ask_qwen_anything(messages, processor, model):

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
    generated_ids = model.generate(**inputs, max_new_tokens=4096)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    response = output_text[0]

    return response

def calculate_score(gt, pred, bbox_norm):

    pred = pred.split("</think>")[-1]

    if gt == pred:
        return 1
    
    else:
        if "complete" in gt:
            print(f"\nError: gt contains 'complete' but does not match pred: {gt} vs {pred}")
            return 0

        content_icon_match = re.search(r"click\s+(.*?)\s+icon", gt)
        sol_icon_match = re.search(r"click\s+(.*?)\s+icon", pred)
        if not content_icon_match:
            print(f"\nError: Unable to extract icon names from gt: {gt}")
            return 0
        if not sol_icon_match:
            print(f"\nError: Unable to extract icon names from pred: {pred}")
            return 0


        content_icon = content_icon_match.group(1).strip()
        sol_icon = sol_icon_match.group(1).strip()

        sol_point_match = re.search(r'\((\d+),(\d+)\)', pred)

        if not sol_point_match:
            print(f"\nError: Unable to extract coordinates from pred: {pred}")
            return 0

        x_min, y_min, x_max, y_max = bbox_norm

        sol_x = int(sol_point_match.group(1))
        sol_y = int(sol_point_match.group(2))

        if sol_x <= x_max and sol_x >= x_min and sol_y <= y_max and sol_y >= y_min and content_icon == sol_icon:    
            return 1

        else:
            return 0


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Evaluate summary img.')
    parser.add_argument('--model_path', type=str, required=True, help='')
    parser.add_argument('--test_file', type=str, required=True, help='')
    parser.add_argument('--save_file', type=str, required=True, help='')

    args = parser.parse_args()

    model_path = args.model_path
    eval_file = args.test_file
    save_file = args.save_file

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
                        else:
                            print(f"Warning: Found a line without 'idx' in {save_file}. Skipping this line for resume check: {line.strip()}")
                    except json.JSONDecodeError:
                        print(f"Warning: Found invalid JSON line in {save_file}. Skipping: {line.strip()}")
            print(f"Found {len(processed_ids)} already processed IDs.")
        except Exception as e:
            print(f"Error reading existing save file {save_file}: {e}. Starting from scratch.")
            processed_ids = set()   
            resuming = False 
    else:
        print(f"Output file {save_file} not found. Starting fresh.")


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

    with open(eval_file, "r", encoding='utf-8') as f:
        datas = json.load(f)


    open_mode = 'a' if resuming else 'w'
    print(f"Starting processing. Results will be saved to {save_file} (mode: {open_mode})")

    items_processed_this_run = 0
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

            content = []
            if 'images' in data and data['images']:
                content.append({
                    "type": "image",
                    "image": data['images'][0]
                })
            else:
                pass

            content.append({
                "type": "text",
                "text": data['messages'][0]['content']
            })

            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": content}
            ]

            try:
                # vllm 
                prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                mm_data = {}
                if image_inputs is not None: mm_data["image"] = image_inputs
                if video_inputs is not None: mm_data["video"] = video_inputs
                llm_inputs = {"prompt": prompt, "multi_modal_data": mm_data}

                outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
                response = outputs[0].outputs[0].text

                data['prediction'] = response

   
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                f_out.flush() 

                items_processed_this_run += 1

       

            except Exception as e:
                print(f"\nError processing idx {idx}: {e}")
               
                continue

    print("\n--- Processing Summary ---")
    print(f"Total items in input file: {len(datas)}")
    print(f"Items skipped (already processed): {items_skipped}")
    print(f"Items processed in this run: {items_processed_this_run}")
    total_in_output = items_skipped + items_processed_this_run
    print(f"Total items expected in output file ({save_file}): {total_in_output}")
    print(f"Processing complete.")


    print("\nScript finished.")