ood_builder.py를 사용해 ood environment 제작 (random seed 42로 통일)

python ood_builder.py \
  --base ui_environment_448/20260212_181922 \
  --out ui_environment_448_ood/20260212_181922 \
  --alt_icons icons_var \
  --noise_icons icons_noise \
  --noise_k 2 \
  --seed 42

generate_dataset_ood.py를 이용해 ood dataset 제작
(여기서 edge도 subtree 4만을 선택해서 제작하였음)

python generate_dataset_ood.py \
  --ood_root ui_environment_448_ood/20260212_181922 \
  --out_dir datas \
  --subtree 4 \
  --max_path_tasks 2200 \
  --max_edge_samples 0 \
  --seed 42\
  --include_system_actions

이후 eval

python eval/inference_qwen2p5_mixed_vllm.py \
    --model_path gelab-sft-448-seed42 \
    --test_file data_engine/datas/ood_test_Base.json \
    --save_file result_Base.json

python eval/eval_ood.py \
  --model_path gelab-sft-448-seed42 \
  --data_dir datas/ood \
  --env_dir data_engine \
  --infer_script eval/inference_qwen2p5_mixed_vllm.py \
  --out_dir results_ood