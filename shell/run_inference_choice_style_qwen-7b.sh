#!/bin/bash

# 定义取值范围
conflict_types=('default' 'fact' 'temporal' 'semantic')
styles=('obj' 'blog' 'news_report' 'science_reference' 'confident_language' 'technical_language')
datasets=('misBen')  # 2WikiMultihopQA  misBen

# Iterate over conflict types, styles, and datasets
for conflict_type in "${conflict_types[@]}"; do
  for style in "${styles[@]}"; do
    for dataset in "${datasets[@]}"; do
      CUDA_VISIBLE_DEVICES=1 python ../src/inference_single_file.py \
        --model_path "/data/miaopeng/workplace/misbench/hugging_cache/Qwen2.5-7B-instruct" \
        --input_file "/data/miaopeng/workplace/misbench/data_inference_prompts/Qwen2.5-0.5B-instruct/$dataset/$conflict_type/${conflict_type}_style_${style}_choice.jsonl" \
        --out_dir "/data/miaopeng/workplace/misbench/inference_results/llama-3-70B-instruct/$dataset/choice"
      done
    done
done