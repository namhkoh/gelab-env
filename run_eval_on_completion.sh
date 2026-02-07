#!/bin/bash
# Monitor ST-RL training and run eval when complete

CHECKPOINT_DIR="./checkpoint/gui_exp/st_rl_448_balanced/v0-20260202_112544"
EVAL_SCRIPT="eval/eval_st_rl.py"

echo "=============================================="
echo "Monitoring ST-RL training completion..."
echo "Checkpoint: $CHECKPOINT_DIR"
echo "=============================================="

# Wait for training process to finish
while pgrep -f "st_rl_448_balanced" > /dev/null; do
    echo "$(date): Training still running..."
    sleep 30
done

echo ""
echo "=============================================="
echo "Training complete! Starting evaluation..."
echo "Time: $(date)"
echo "=============================================="

# Find the final checkpoint
FINAL_CHECKPOINT=$(ls -d ${CHECKPOINT_DIR}/checkpoint-* 2>/dev/null | sort -V | tail -1)
echo "Final checkpoint: $FINAL_CHECKPOINT"

# Run evaluation
echo ""
echo "Running interactive evaluation (Paper Table 2 style)..."
python eval/eval_interactive.py \
    --model_path "$FINAL_CHECKPOINT" \
    --test_data "datas/448_retrain/test_st_rl.json" \
    --env_dir "datas/448_retrain" \
    --output_dir "eval_results/st_rl_balanced_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "=============================================="
echo "Evaluation complete!"
echo "Time: $(date)"
echo "=============================================="
