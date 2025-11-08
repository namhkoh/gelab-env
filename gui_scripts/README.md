# Training Scripts Guide

This repository contains various training scripts for different stages of model training.

## 🛠️ Installation

To install from source:
```shell
pip install -e .
```

Running Environment:

|              | Range        | Recommended | Notes                                     |
| ------------ |--------------| ----------- | ----------------------------------------- |
| python       | >=3.9        | 3.10        |                                           |
| cuda         |              | cuda12      | No need to install if using CPU, NPU, MPS |
| torch        | >=2.0        |             |                                           |
| transformers | >=4.33       | 4.51      |                                           |
| modelscope   | >=1.23       |             |                                           |
| peft | >=0.11,<0.16 | ||
| trl | >=0.13,<0.18 | 0.17 |RLHF|
| deepspeed    | >=0.14       | 0.14.5 | Training                                  |
| vllm         | >=0.5.1      | 0.7.3/0.8       | Inference/Deployment/Evaluation           |
| lmdeploy     | >=0.5        | 0.8       | Inference/Deployment/Evaluation           |
| evalscope | >=0.11       |  | Evaluation |


## Available Scripts

### SFT (Supervised Fine-Tuning)
```bash
gui_scripts/sft.sh
```
This script performs supervised fine-tuning on the model using labeled data.

### Single-Turn RL Training
```bash
gui_scripts/single_turn_rl.sh
```
This script executes reinforcement learning training in a single-turn conversation setting.

### Multi-Turn RL Training
```bash
gui_scripts/multi_turn_rl.sh
```
This script implements reinforcement learning training in a multi-turn conversation context.

## Usage

1. Make sure you have all the required dependencies installed
2. Ensure the scripts have executable permissions:
   ```bash
   chmod +x gui_scripts/*.sh
   ```
3. Run the desired script from the repository root:
   ```bash
   ./gui_scripts/sft.sh
   # or
   ./gui_scripts/single_turn_rl.sh
   # or
   ./gui_scripts/multi_turn_rl.sh
   ```

## Note
- Please make sure all required data and model checkpoints are properly set up before running the training scripts
- Monitor the training logs for any potential issues or errors
- For detailed configuration options, please check the individual script files


## Evaluation

The evaluation process consists of two main steps:

### 1. Generate Inference Results
To generate model predictions:
```bash
python3 eval/inference_qwen2p5_mixed_vllm.py \
    --model_path <path_to_your_model> \
    --test_file <path_to_test.json> \
    --savefile result.json
```

Parameters:
- `model_path`: Path to your trained model
- `test_file`: Path to the test dataset in JSON format
- `savefile`: Output file to save the inference results (default: result.json)

### 2. Calculate Evaluation Metrics
To compute the evaluation scores:
```bash
python3 eval/calculate_score_refine.py --file result.json
```

Parameters:
- `file`: Path to the inference results file generated in step 1

The script will output various metrics to assess the model's performance.

Note: Ensure you have all required dependencies installed before running the evaluation scripts.