#!/bin/bash
set -e
source activate gelab
cd /home/irteam/gelab-env

# Wait for training to finish (check for swift/torch processes)
echo "Waiting for training to complete..."
while ps aux | grep -E "swift sft|torch.distributed" | grep -v grep > /dev/null 2>&1; do
    sleep 60
done
echo "Training finished. Starting eval..."

# Find the latest checkpoint
CKPT_DIR=$(ls -dt ./checkpoint/gui_exp/sft_amex/v0-*/checkpoint-* 2>/dev/null | head -1)
if [ -z "$CKPT_DIR" ]; then
    echo "ERROR: No checkpoint found!"
    exit 1
fi
echo "Using checkpoint: $CKPT_DIR"

CACHE_DIR="/home/irteam/data-vol1/.cache/huggingface/datasets"
OUT_DIR="./eval_results"
mkdir -p "$OUT_DIR"

echo ""
echo "============================================"
echo "1/4: Running eval_screenspot.py"
echo "============================================"
python eval/eval_screenspot.py \
    --model_path "$CKPT_DIR" \
    --num_gpus 4 \
    --cache_dir "$CACHE_DIR" \
    --output_file "$OUT_DIR/eval_screenspot_sft_amex.json"

echo ""
echo "============================================"
echo "2/4: Running eval_motif.py"
echo "============================================"
python eval/eval_motif.py \
    --model_path "$CKPT_DIR" \
    --num_gpus 4 \
    --cache_dir "$CACHE_DIR" \
    --output_file "$OUT_DIR/eval_motif_sft_amex.json"

echo ""
echo "============================================"
echo "3/4: Running eval_vwb_ag.py"
echo "============================================"
python eval/eval_vwb_ag.py \
    --model_path "$CKPT_DIR" \
    --num_gpus 4 \
    --cache_dir "$CACHE_DIR" \
    --output_file "$OUT_DIR/eval_vwb_ag_sft_amex.json"

echo ""
echo "============================================"
echo "4/4: Running eval_vwb_eg.py"
echo "============================================"
python eval/eval_vwb_eg.py \
    --model_path "$CKPT_DIR" \
    --num_gpus 4 \
    --cache_dir "$CACHE_DIR" \
    --output_file "$OUT_DIR/eval_vwb_eg_sft_amex.json"

echo ""
echo "============================================"
echo "ALL EVALS COMPLETE"
echo "============================================"
for f in "$OUT_DIR"/eval_*_sft_amex.json; do
    echo "$(basename $f): $(cat $f | python3 -c 'import sys,json; d=json.load(sys.stdin); print(f"accuracy={d.get(\"accuracy\",\"N/A\"):.2f}%")')"
done
