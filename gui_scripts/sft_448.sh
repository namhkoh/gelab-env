#!/bin/bash
# SFT Training with 448x448 images - aligned with paper
# Paper specs: max_pixels=200704, lr=1e-5, batch=2, grad_accum=2, epochs=1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# WandB config - use environment variable if available, otherwise disable
if [ -z "$WANDB_API_KEY" ]; then
    echo "WANDB_API_KEY not set, disabling wandb reporting"
    REPORT_TO="none"
else
    export WANDB_PROJECT=gelab
    REPORT_TO="wandb"
fi

# Number of GPUs
NPROC_PER_NODE=8

# Dataset path (448x448 format)
DATASET_PATH="datas/448/sft_448.json"

# Output directory
OUTPUT_DIR="checkpoint/gui_exp/sft_448/v1-$(date +%Y%m%d-%H%M%S)"

# Run training with deepspeed
PYTHONPATH=. torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_port=29501 \
    swift/cli/sft.py \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --train_type full \
    --dataset "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --max_length 2048 \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --eval_steps 100 \
    --save_steps 100 \
    --logging_steps 10 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --save_total_limit 2 \
    --deepspeed zero2 \
    --dataloader_num_workers 4 \
    --max_pixels 200704 \
    --report_to $REPORT_TO

echo "Training complete. Model saved to: $OUTPUT_DIR"
