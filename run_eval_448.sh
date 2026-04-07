#!/bin/bash
set -e
source activate gelab
cd /home/irteam/gelab-env

CKPT_DIR="/home/irteam/gelab-env/checkpoint/gui_exp/sft_amex/v0-20260404_141926_448/checkpoint-2473"
CACHE_DIR="/home/irteam/data-vol1/.cache/huggingface/datasets"
OUT_DIR="./eval_results"
SUFFIX="sft_amex_v0404"
mkdir -p "$OUT_DIR"

echo "Using checkpoint: $CKPT_DIR"

echo ""
echo "============================================"
echo "1/4: Running eval_screenspot.py"
echo "============================================"
python eval/eval_screenspot.py \
    --model_path "$CKPT_DIR" \
    --num_gpus 4 \
    --cache_dir "$CACHE_DIR" \
    --output_file "$OUT_DIR/eval_screenspot_${SUFFIX}.json"

echo ""
echo "============================================"
echo "2/4: Running eval_motif.py"
echo "============================================"
python eval/eval_motif.py \
    --model_path "$CKPT_DIR" \
    --num_gpus 4 \
    --cache_dir "$CACHE_DIR" \
    --output_file "$OUT_DIR/eval_motif_${SUFFIX}.json"

echo ""
echo "============================================"
echo "3/4: Running eval_vwb_ag.py"
echo "============================================"
python eval/eval_vwb_ag.py \
    --model_path "$CKPT_DIR" \
    --num_gpus 4 \
    --cache_dir "$CACHE_DIR" \
    --output_file "$OUT_DIR/eval_vwb_ag_${SUFFIX}.json"

echo ""
echo "============================================"
echo "4/4: Running eval_vwb_eg.py"
echo "============================================"
python eval/eval_vwb_eg.py \
    --model_path "$CKPT_DIR" \
    --num_gpus 4 \
    --cache_dir "$CACHE_DIR" \
    --output_file "$OUT_DIR/eval_vwb_eg_${SUFFIX}.json"

echo "All evals complete!"
