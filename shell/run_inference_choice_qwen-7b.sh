#!/bin/bash

conflict_types=('default' 'fact' 'temporal' 'semantic')
datasets=('misBen')  # 2WikiMultihopQA

# Iterate over conflict types and datasets
for conflict_type in "${conflict_types[@]}"; do
  for dataset in "${datasets[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python ../src/inference_single_file.py \
      --model_path "/data/miaopeng/workplace/misbench/hugging_cache/Qwen2.5-7B-instruct" \
      --input_file "/data/miaopeng/workplace/misbench/data_inference_prompts/Qwen2.5-0.5B-instruct/$dataset/$conflict_type/${conflict_type}_style_obj_choice.jsonl" \
      --out_dir "/data/miaopeng/workplace/misbench/inference_results/llama-3-70B-instruct/$dataset/choice"
  done
done