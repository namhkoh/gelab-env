import argparse
import os
import glob
import json
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import traceback

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

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def resolve_image_path(img_path: str, env_dir: str, test_file: str) -> str:
    """
    img_path가 상대경로일 때, 우선 env_dir 기준으로 붙여보고,
    없으면 test_file 디렉토리 기준으로도 붙여본다.
    """
    if not img_path:
        return img_path

    # already absolute
    if os.path.isabs(img_path):
        return img_path

    # 1) env_dir 기준
    cand = os.path.normpath(os.path.join(os.path.abspath(env_dir), img_path))
    if os.path.exists(cand):
        return cand

    # 2) test_file 위치 기준 (datas/... 안에 상대경로가 같이 들어있는 경우)
    base_dir = os.path.dirname(os.path.abspath(test_file))
    cand2 = os.path.normpath(os.path.join(base_dir, img_path))
    if os.path.exists(cand2):
        return cand2

    # 3) 마지막으로 CWD 기준(디버깅용)
    cand3 = os.path.normpath(os.path.join(os.getcwd(), img_path))
    if os.path.exists(cand3):
        return cand3

    # 못 찾으면 원래 값 반환 (에러 메시지에 원인 드러나게)
    return img_path

def read_processed_ids(save_file: str):
    processed_ids = set()
    resuming = False
    if os.path.exists(save_file):
        resuming = True
        try:
            with open(save_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if "idx" in rec:
                            processed_ids.add(rec["idx"])
                    except json.JSONDecodeError:
                        continue
        except Exception:
            processed_ids = set()
            resuming = False
    return processed_ids, resuming

def collect_test_files(data_dir: str):
    # datas/ood 안의 json들 중 ood_test_*.json 전부
    files = glob.glob(os.path.join(data_dir, "ood_test_*.json"))
    files = [f for f in files if os.path.isfile(f)]
    files.sort()
    return files

def run_one_file(llm: LLM, sampling_params: SamplingParams, processor: AutoProcessor,
                 test_file: str, save_file: str, env_dir: str):

    processed_ids, resuming = read_processed_ids(save_file)
    mode = "a" if resuming else "w"

    print(f"\n=== RUN ===")
    print(f"test: {test_file}")
    print(f"save: {save_file} (mode={mode})")
    print(f"resume={resuming}, already={len(processed_ids)}")

    with open(test_file, "r", encoding="utf-8") as f:
        datas = json.load(f)

    skipped = 0
    done = 0

    with open(save_file, mode, encoding="utf-8") as f_out:
        for data in tqdm(datas, desc=os.path.basename(test_file)):
            if "idx" not in data:
                continue
            idx = data["idx"]
            if idx in processed_ids:
                skipped += 1
                continue

            content = []
            if "images" in data and data["images"]:
                img_path = resolve_image_path(data["images"][0], env_dir=env_dir, test_file=test_file)
                content.append({"type": "image", "image": img_path})

            # 원본 코드랑 동일하게 messages[0]['content'] 사용
            content.append({"type": "text", "text": data["messages"][0]["content"]})

            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ]

            try:
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

                data["prediction"] = response
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                f_out.flush()

                done += 1

            except Exception as e:
                print(f"\nError processing idx {idx}: {e}")
                # print(traceback.format_exc())
                continue

    print(f"--- summary: total={len(datas)}, skipped={skipped}, processed_now={done}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_dir", required=True, help="예: datas/ood")
    parser.add_argument("--env_dir", required=True, help="예: data_engine (이미지 상대경로 prefix 기준)")
    parser.add_argument("--infer_script", default="", help="호환용으로 남겨둠(실제로는 이 파일이 inference까지 담당)")
    parser.add_argument("--out_dir", required=True)

    # vLLM 메모리 안전 옵션
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.75)
    parser.add_argument("--max_model_len", type=int, default=8192)

    # resume 관련
    parser.add_argument("--skip_existing", action="store_true", help="결과 파일이 있으면 해당 test 전체 스킵")

    args = parser.parse_args()

    ensure_dir(args.out_dir)

    test_files = collect_test_files(args.data_dir)
    if not test_files:
        raise RuntimeError(f"No test files found: {args.data_dir}/ood_test_*.json")

    # 모델은 1번만 로드
    llm = LLM(
        model=args.model_path,
        limit_mm_per_prompt={"image": 1, "video": 0},
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    sampling_params = SamplingParams(
        temperature=0,
        top_p=0.001,
        repetition_penalty=1.05,
        max_tokens=1024,
        stop_token_ids=[],
    )

    processor = AutoProcessor.from_pretrained(args.model_path)

    # 15개 파일 순회
    for tf in test_files:
        base = os.path.splitext(os.path.basename(tf))[0]  # ood_test_Base...
        save_file = os.path.join(args.out_dir, f"result_{base}.jsonl")

        if args.skip_existing and os.path.exists(save_file):
            print(f"[SKIP existing] {save_file}")
            continue

        run_one_file(llm, sampling_params, processor, tf, save_file, env_dir=args.env_dir)

    print("\nALL DONE.")

if __name__ == "__main__":
    main()
