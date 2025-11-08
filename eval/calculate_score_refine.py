import json
import re
# import math
import os
import traceback
from tqdm import tqdm
import argparse
from collections import defaultdict


def calculate_score(gt, pred, bbox_norm):
    pred = pred.split("</think>")[-1]

    if "complete" in gt and "complete" in pred:
        return 1
    
    else:
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

parser = argparse.ArgumentParser(description='Evaluate summary img, resuming if output file exists, and calculate scores.')
parser.add_argument('--file', type=str, required=True, help='Path to the input test JSONL file')
args = parser.parse_args()

save_file = args.file


# --- Scoring Logic ---
print(f"\n--- Starting Score Calculation from {save_file} ---")

score_list = []
score_list_wo = []
score_list_com = []
score_list_edge = []
score_list_icon_cap = []
score_list_icon_gnd = []
scoring_errors = 0
items_scored = 0

score_path_1= []
score_path_2= []
score_path_3= []
score_path_4= []
score_path_5= []
score_path_6= []
score_path_7= []

score_path1_dict = defaultdict(int)
score_path2_dict = defaultdict(int)
score_path3_dict = defaultdict(int)
score_path4_dict = defaultdict(int)
score_path5_dict = defaultdict(int)
score_path6_dict = defaultdict(int)
score_path7_dict = defaultdict(int)

if os.path.exists(save_file):
    if save_file.endswith(".json"):
        print(f"Warning: Save file {save_file} has a .json extension. Skipping scoring.")
        # Optional: You could still read and process the file if needed, but it's skipped here.
        with open(save_file, "r", encoding='utf-8') as f_score:
            datas = json.load(f_score)

    else:
        with open(save_file, "r", encoding='utf-8') as f_score:
            datas = [json.loads(line) for line in f_score.readlines()]  

    for line_num, data in enumerate(tqdm(datas, desc="Calculating Scores")):
        try:

            items_scored += 1 # Count attempts to score

            # --- Check for necessary keys ---
            if "prediction" not in data:
                print(f"Warning: Line {line_num+1} missing 'prediction'. Cannot score.")
                scoring_errors += 1
                continue
            if not ("messages" in data and isinstance(data["messages"], list) and len(data["messages"]) > 1 and "content" in data["messages"][1]):
                print(f"Warning: Line {line_num+1} missing valid ground truth in 'messages[1][content]'. Cannot score.")
                scoring_errors += 1
                continue
            if "task" not in data and not ("Null" in data.get("messages", [{}])[0].get("content", "") or data.get("messages", [{}, {}])[1].get("content") == "explain: this is target page.\tAction: complete"):
                # If not icon task and doesn't match com/edge patterns, we might miss categorizing it if 'task' is missing.
                # Depending on requirements, you might assign a default task or skip.
                print(f"Warning: Line {line_num+1} missing 'task' key, categorization might be incomplete.")
                # Assign a default or skip based on your needs, here we'll proceed but it won't match icon tasks.


            # --- Extract data for scoring ---
            bbox_norm = data.get("bbox_norm") # Use .get for optional key
            gt = data["messages"][1]["content"]
            pred = data["prediction"]

            # --- Calculate score ---
            try:
                score = calculate_score(gt, pred, bbox_norm)
            except Exception as e:
                print(f"Error calculating score for line {line_num+1}: {e}")
                scoring_errors += 1
                continue # Skip item if score calculation fails

            # --- Categorize score ---
            path = data.get("path", "")
            task = data.get("task", "").lower() # Default to empty string if task missing

            if path == 1:
                score_path_1.append(score)
            elif path == 2:
                score_path_2.append(score)
            elif path == 3:
                score_path_3.append(score)
            elif path == 4:
                score_path_4.append(score)
            elif path == 5:
                score_path_5.append(score)
            elif path == 6:
                score_path_6.append(score)
            elif path == 7:
                score_path_7.append(score)

            if path == 1:
                score_path1_dict[task] += score
            elif path == 2:
                score_path2_dict[task] += score
            elif path == 3:
                score_path3_dict[task] += score
            elif path == 4:
                score_path4_dict[task] += score
            elif path == 5:
                score_path5_dict[task] += score
            elif path == 6:
                score_path6_dict[task] += score
            elif path == 7:
                score_path7_dict[task] += score


            if task == "caption":
                score_list_icon_cap.append(score)
            elif task == "grounding":
                score_list_icon_gnd.append(score)
            else:
                # Apply to general list and specific categories if applicable
                score_list.append(score)

                com = False
                edge = False
                # Check for 'complete' task pattern
                if gt == "Explain: this is target page.\tAction: complete" or gt == "explain: this is target page.\tAction: complete":
                    com = True
                    score_list_com.append(score)

                # Check for 'edge' case pattern in user prompt ('messages[0]')
                user_prompt_content = data.get("messages", [{}])[0].get("content", "")
                # Handle both string and list content for user prompt flexibility
                is_edge_content = False
                if isinstance(user_prompt_content, str) and "Null" in user_prompt_content:
                    is_edge_content = True
                elif isinstance(user_prompt_content, list):
                    for item in user_prompt_content:
                        if isinstance(item, dict) and item.get("type") == "text" and "Null" in item.get("text", ""):
                            is_edge_content = True
                            break
                if is_edge_content:
                    edge = True
                    score_list_edge.append(score)

                # Add to 'wo' (without complete/edge) if neither flag is set
                if not com and not edge:
                    score_list_wo.append(score)

        except json.JSONDecodeError:
            print(f"Warning: Invalid JSON on line {line_num+1} during scoring. Skipping: {line.strip()}")
            scoring_errors += 1
        except (KeyError, IndexError, TypeError) as e:
            print(f"Error accessing data on line {line_num+1} during scoring: {e}. Skipping item.")
            # traceback.print_exc() # Uncomment for details
            scoring_errors += 1

    print(f"\nFinished scoring. Attempted to score {items_scored} items. Encountered {scoring_errors} errors during scoring.")

    # --- Print Score Results ---
    def print_scores(name, scores):
        if scores:
            total = len(scores)
            correct = sum(s for s in scores if isinstance(s, (int, float))) # Sum numeric scores
            avg_score = round(correct / total * 100, 3) if total > 0 else 0
            print(f"================== \n{name}:")
            print(f"score: {avg_score}")
            print(f"total: {total}")
            print(f"correct (sum of scores): {correct}") # Changed label for clarity if score isn't binary 0/1
            return avg_score
        else:
            print(f"================== \n{name}: No data.")
            return None

    all_score = print_scores("all (non-icon)", score_list)
    wo_score = print_scores("w/o complete & edge", score_list_wo)
    com_score = print_scores("only complete", score_list_com)
    edge_score = print_scores("only edge", score_list_edge)

    # Print summary line only if all relevant scores were calculated
    if all(s is not None for s in [all_score, wo_score, com_score, edge_score]):
        print(f"================== \ntotal:\n{all_score} -- {wo_score} -- {com_score} -- {edge_score}")

    cap_score = print_scores("icon cap", score_list_icon_cap)
    gnd_score = print_scores("icon gnd", score_list_icon_gnd)


    # print()
    print_scores("path 1", score_path_1)
    print_scores("path 2", score_path_2)
    print_scores("path 3", score_path_3)
    print_scores("path 4", score_path_4)
    print_scores("path 5", score_path_5)
    print_scores("path 6", score_path_6)
    print_scores("path 7", score_path_7)

    def score_dict(path, score_dict):
        count = 0
        average_list = []
        for k, v in score_dict.items():
            if isinstance(v, (int, float)):
                if v == path + 1:
                    count += 1
                average_list.append(v)

        print(f"path {path}, task total {len(score_dict)}")
        print(f"Average steps: {sum(average_list)/len(average_list)}")
        print(f"Overall accuracy: {round(count/len(score_dict), 3)}\n")
        return round(count/len(score_dict), 3)

    print(f"=====================================")
    score_dict(1, score_path1_dict)
    score_dict(2, score_path2_dict)
    score_dict(3, score_path3_dict)
    score_dict(4, score_path4_dict)
    score_dict(5, score_path5_dict)
    score_dict(6, score_path6_dict)
    score_dict(7, score_path7_dict)

    # print(f"score_path2_dict: {score_path2_dict}")

print("\nScript finished.")