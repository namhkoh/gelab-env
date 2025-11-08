import json
import re
import os 
from PIL import Image
from copy import copy, deepcopy

def build_page_transitions(json_data):
    transitions = {}
    
    def process_node(node):
        # Get current page name
        page = node.get("image", "")
        if not page:
            return
            
        # Initialize transitions with three empty lists for this page
        transitions[page] = {
            "bboxes": [],    # List of bbox coordinates
            "targets": [],   # List of corresponding target pages
            "icon_names": []      # List of transition button names
        }
        
        # Add all transitions from this page
        for transition in node.get("transitions", []):
            target = transition.get("target_page")
            bbox = transition.get("icon_bbox")
            name = transition.get("action", "")  # Get transition name
            if target and bbox:
                transitions[page]["bboxes"].append(bbox)
                transitions[page]["targets"].append(target)
                transitions[page]["icon_names"].append(name)
                
        # Recursively process subnodes
        for subnode in node.get("subnodes", []):
            process_node(subnode)
    
    # Start processing from root
    process_node(json_data["root"])
    
    return transitions

def get_normalized_bbox(bbox, image_width, image_height):

    x_min, y_min, x_max, y_max = bbox
    normalized_x_min = round((x_min / image_width) * 1000.0)
    normalized_y_min = round((y_min / image_height) * 1000.0)
    normalized_x_max = round((x_max / image_width) * 1000.0)
    normalized_y_max = round((y_max / image_height) * 1000.0)

    return [normalized_x_min, normalized_y_min, normalized_x_max, normalized_y_max]

def check_math_result_and_give_tips(inputs):
    from .orm import MathAccuracy
    acc = MathAccuracy()
    # a trick
    prompt = 'But wait... It seems I made a mistake,'
    contents = [input['messages'][-1]['content'] for input in inputs]
    rewards = acc(contents, [input['solution'] for input in inputs])
    for reward, input in zip(rewards, inputs):
        content = input['messages'][-1]['content']
        if reward < 1 and prompt not in content:
            if '<answer>' in content:
                content = content[:content.index('<answer>')]
            if '</think>' in content:
                content = content[:content.index('</think>')]
            content += prompt
            input['messages'][-1]['content'] = content
            input['finished'] = False
        else:
            input['finished'] = True
    return inputs


def check_math_result_and_give_tips_multi_turn(inputs):
    from .orm import MathAccuracy
    acc = MathAccuracy()
    prompt = 'The answer is not correct, It seems You made a mistake, you need to recheck very carefully.'
    contents = [input['messages'][-1]['content'] for input in inputs]
    rewards = acc(contents, [input['solution'] for input in inputs])
    for reward, input in zip(rewards, inputs):
        content = input['messages'][-2]['content']
        if reward < 1 and prompt not in content:
            input['messages'].append({'role': 'user', 'content': prompt})
            input['finished'] = False
        else:
            input['finished'] = True
    return inputs

def load_image_dimensions(image_path):

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return width, height
    except Exception as e:
        raise IOError(f"Error opening or reading image '{image_path}': {e}") from e

def check_gelab_result_and_give_tips_multi_turn(inputs, num_turns):

    path = "environment/demo/ui_structure.json"

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    pages = data['pages']


    env_data = json.load(open('environment/demo/ui_structure_layer_fixed.json'))

    page_transitions = build_page_transitions(env_data)
    outputs = []
    for input in inputs:

        input['finished'] = False

        curr_action = input['messages'][-1]['content']

        # '''begin'''
        prompt = input['messages'][0]['content']

        page_identifier_pattern = r"((?:[A-Za-z0-9_]+_)?page_\d+)"


        regex_pattern = rf'From\s+{page_identifier_pattern}\s+to\s+{page_identifier_pattern}'

        match = re.search(regex_pattern, prompt, re.IGNORECASE) # re.IGNORECASE makes "From", "to" case insensitive

        if match:

            target_page = match.group(2)   # Second page identifier
        else:
            print(f"No target_page match found in prompt: {prompt}")
            target_page = "None"

        last_icon_name_match = re.search(r"click\s+(.*?)\s+icon", curr_action)
        last_page_name_match = re.search(r'on\s+((?:[A-Za-z0-9_]+_)?page_\d+)', curr_action, re.IGNORECASE)


        if last_icon_name_match and last_page_name_match:
            last_icon_name = last_icon_name_match.group(1).strip()
            last_page_name = last_page_name_match.group(1).strip()
            # last_page_name = "page_" + last_page_name_match.group(1).strip()

            if last_page_name in pages:
                current_page = None
                for item in pages[last_page_name]["transitions"]:
                    if last_icon_name == item['action']:
                        current_page = item['target_page']
                        break

                if current_page == target_page:     
                    # print(f"The action is correct, curr_action is {curr_action}.")
                    input['finished'] = True
                    input['curr_page'] = input['images'][0]['path'].split('/')[-1].replace('.png', '')
                    input['finish_reason'] = 'stop'
                    outputs.append(deepcopy(input))
                    continue

        if num_turns >= 8:
            input['finished'] = True
            input['curr_page'] = input['images'][0]['path'].split('/')[-1].replace('.png', '')
            input['finish_reason'] = 'stop'
            outputs.append(deepcopy(input))
            continue

        curr_image_path = input['images'][0]['path']
        image_name = curr_image_path.split('/')[-1]
        transitions = page_transitions[image_name]
        curr_action = input['messages'][-1]['content']

        content_point_match = re.search(r'\((\d+),\s*(\d+)\)', curr_action)
        # Position clicked by actor
        try:
            content_x = int(content_point_match.group(1))
            content_y = int(content_point_match.group(2))
            # Clickable bboxes on the image, after normalization
            image_width, image_height = load_image_dimensions(curr_image_path.replace('pages_resize_448', 'pages'))
            normalized_bbox = [get_normalized_bbox(bbox, image_width, image_height) for bbox in transitions['bboxes']]
            click_idx = -1
            # Iterate through clickable bboxes to find where the actor actually clicked
            for idx, bbox in enumerate(normalized_bbox):
                x_min, y_min, x_max, y_max = bbox
                if x_min <= content_x <= x_max and y_min <= content_y <= y_max:
                    click_idx = idx

        except:

            click_idx = -1

        if click_idx != -1:

            new_image_name = transitions['targets'][click_idx] + ".png"
            new_image_path = curr_image_path.replace(image_name, new_image_name)
            input['images'] = [{'bytes': None, 'path': new_image_path}]

            last_message = input['messages'][0]['content']  # Changed to first message
            # Modified regex to match both formats
            history_match = re.search(r'History: (.*?)(?:\. State:|$)', last_message)
            history = history_match.group(1) if history_match else ""
            try:
                clicked_icon_name = curr_action.split(" ")[2]
            except Exception as e:
                clicked_icon_name = "None"
                print("Error parsing clicked icon name:", e)

            # Count existing steps to determine next step number
            step_count = len(re.findall(r'step\d+:', history))
            next_step = f"step{step_count + 1}: click {clicked_icon_name} icon on {image_name[:-4]}"

            new_history = f"{history}.{next_step}" if history else next_step
            
            # Create new instruction message with original format
            instruction_parts = last_message.split(' History: ')
            base_instruction = instruction_parts[0]  # Gets the "Instruction: from page_X to page_Y" part
            new_message = f'{base_instruction} History: {new_history}'.replace("Null.", "")
            # print(input['messages'][0]['content'])
            # print(type(input['messages'][0]['content']))
            input['messages'] = [{'role': 'user', 'content': new_message}]

            outputs.append(deepcopy(input))

        else:
            last_message = input['messages'][0]['content']  # Changed to first message
            # Modified regex to match both formats
            history_match = re.search(r'History: (.*?)(?:\. State:|$)', last_message)
            history = history_match.group(1) if history_match else "None"

            try:    
                # clicked_icon_name = curr_action.split(" ")[1]
                clicked_icon_name = curr_action.split(" ")[2]
            except:
                clicked_icon_name = "None"
            
            # Count existing steps to determine next step number
            step_count = len(re.findall(r'step\d+:', history))
            next_step = f"step{step_count + 1}: click {clicked_icon_name} icon on {image_name[:-4]}"
            new_history = f"{history}.{next_step}" if history else next_step
            
            # Create new instruction message with original format
            instruction_parts = last_message.split(' History: ')
            base_instruction = instruction_parts[0]  # Gets the "Instruction: from page_X to page_Y" part
            new_message = f'{base_instruction} History: {new_history}'.replace("Null.", "")

            # input['messages'] = [{'role': 'user', 'content': new_message}]
            if input['messages'] and len(input['messages']) > 0:
                input['messages'][0]['content'] = new_message
            else:
                input['messages'] = [{'role': 'user', 'content': new_message}]
            outputs.append(deepcopy(input))

    return outputs  

def gelab_multi_turn_wo_complete(inputs, num_turns):

    path = "environment/demo/ui_structure.json"
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    pages = data['pages']


    env_data = json.load(open('environment/demo/ui_structure_layer_fixed.json'))
    page_transitions = build_page_transitions(env_data)
    outputs = []
    for input in inputs:

        input['finished'] = False

        curr_action = input['messages'][-1]['content']

        prompt = input['messages'][0]['content']
        target_page_match = re.search(r'to page_(\d+)', prompt)
        target_page = "page_" + target_page_match.group(1)
        try:
            last_icon_name_match = re.search(r"click\s+(.*?)\s+icon", curr_action)
            last_page_name_match = re.search(r"page_(\d+)", curr_action)
            last_icon_name = last_icon_name_match.group(1).strip()
            last_page_name = "page_" + last_page_name_match.group(1).strip()
            current_page = None
            for item in pages[last_page_name]["transitions"]:
                if last_icon_name == item['action']:
                    current_page = item['target_page']
                    break
        except:
            current_page = None

        if current_page == target_page:     
            print(f"The action is correct, curr_action is {curr_action}.")
            input['finished'] = True
            input['curr_page'] = input['images'][0]['path'].split('/')[-1].replace('.png', '')
            input['finish_reason'] = 'stop'
            outputs.append(deepcopy(input))
            continue

        else:
            if num_turns >= 8:
                input['finished'] = True
                input['curr_page'] = input['images'][0]['path'].split('/')[-1].replace('.png', '')
                input['finish_reason'] = 'stop'
                outputs.append(deepcopy(input))
                continue

            curr_image_path = input['images'][0]['path']
            image_name = curr_image_path.split('/')[-1]
            transitions = page_transitions[image_name]
            curr_action = input['messages'][-1]['content']

            content_point_match = re.search(r'\((\d+),\s*(\d+)\)', curr_action)
            # Position clicked by actor
            try:
                content_x = int(content_point_match.group(1))
                content_y = int(content_point_match.group(2))
                # Clickable bboxes on the image, after normalization
                image_width, image_height = load_image_dimensions(curr_image_path.replace('pages_resize_448', 'pages'))
                normalized_bbox = [get_normalized_bbox(bbox, image_width, image_height) for bbox in transitions['bboxes']]
                click_idx = -1
                # Iterate through clickable bboxes to find where the actor actually clicked
                for idx, bbox in enumerate(normalized_bbox):
                    x_min, y_min, x_max, y_max = bbox
                    if x_min <= content_x <= x_max and y_min <= content_y <= y_max:
                        click_idx = idx

            except:
                click_idx = -1

            if click_idx != -1:

                new_image_name = transitions['targets'][click_idx] + ".png"
                new_image_path = curr_image_path.replace(image_name, new_image_name)
                input['images'] = [{'bytes': None, 'path': new_image_path}]

                last_message = input['messages'][0]['content']  # Changed to first message
                # Modified regex to match both formats
                history_match = re.search(r'History: (.*?)(?:\. State:|$)', last_message)
                history = history_match.group(1) if history_match else ""

                try:
                    clicked_icon_name = curr_action.split(" ")[1]
                except Exception as e:
                    clicked_icon_name = "None"
                    print("Error parsing clicked icon name:", e)

                # Count existing steps to determine next step number
                step_count = len(re.findall(r'step\d+:', history))
                next_step = f"step{step_count + 1}: click {clicked_icon_name} icon on {image_name[:-4]}"
                # Combine existing history with new step
                new_history = f"{history}.{next_step}" if history else next_step

                # Create new instruction message with original format
                instruction_parts = last_message.split(' History: ')
                base_instruction = instruction_parts[0]  # Gets the "Instruction: from page_X to page_Y" part
                new_message = f'{base_instruction} History: {new_history}'.replace("Null.", "")
                # print(input['messages'][0]['content'])
                # print(type(input['messages'][0]['content']))
                input['messages'] = [{'role': 'user', 'content': new_message}]

                outputs.append(deepcopy(input))

            else:
                last_message = input['messages'][0]['content']  # Changed to first message

                # Modified regex to match both formats
                history_match = re.search(r'History: (.*?)(?:\. State:|$)', last_message)
                history = history_match.group(1) if history_match else ""
                try:    
                    clicked_icon_name = curr_action.split(" ")[1]
                except:
                    clicked_icon_name = "None"

                # Count existing steps to determine next step number
                step_count = len(re.findall(r'step\d+:', history))
                next_step = f"step{step_count + 1}: click {clicked_icon_name} icon on {image_name[:-4]}"
                new_history = f"{history}.{next_step}" if history else next_step

                # Create new instruction message with original format
                instruction_parts = last_message.split(' History: ')
                base_instruction = instruction_parts[0]  # Gets the "Instruction: from page_X to page_Y" part
                new_message = f'{base_instruction} History: {new_history}'.replace("Null.", "")

                input['messages'] = [{'role': 'user', 'content': new_message}]

                outputs.append(deepcopy(input))

    return outputs


multi_turns = {
    'math_tip_trick': check_math_result_and_give_tips,
    'math_tip_trick_multi_turn': check_math_result_and_give_tips_multi_turn,
    'gelab_multi_turn': check_gelab_result_and_give_tips_multi_turn,
    'gelab_multi_turn_wo_complete' : gelab_multi_turn_wo_complete,
}
