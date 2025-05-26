import json
import os
import argparse
import torch
import glob
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import numpy as np

"""
inference on a single file
the evaluation is based on the question-answering format
5 choices
"""

def inference(model_name, input_file, out_dir):
    """
    Run inference using the specified model on the input JSONL file and save the results.
    :param model_name: str, name of the pre-trained model
    :param input_file: str, path to the input JSONL file
    :param out_dir: str, directory to save the output JSON file
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=False, use_fast=False, trust_remote_code=True)
    model_type = model_name.split("/")[-1]
    print(f"Accessing {torch.cuda.device_count()} GPUs!")
    llm = LLM(model=model_name, dtype="float16" if model_type != "gemma-2-27b-it" else "bfloat16", tensor_parallel_size=torch.cuda.device_count(), trust_remote_code=True, max_logprobs=1000)
    sampling_params = SamplingParams(temperature=0, logprobs=1000)
    
    output_dir = os.path.join(out_dir, os.path.basename(model_name))
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, os.path.basename(input_file))
    
    prompts = []
    all_datas = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            prompts.append(data['prompt'])
            all_datas.append(data)
    
    print(f"Running inference on {len(prompts)} prompts...")
    
    for i in range(0, len(all_datas), 20000):
        batch_prompts = prompts[i:i + 20000]
        output = llm.generate(batch_prompts, sampling_params)
    
        predictions = []
        probs = []
        prediction_texts = []
        for num in range(len(output)):
            # Save the prediction text
            tmp_output_text = output[num].outputs[0].text
            if "assistant" in tmp_output_text:
                try:
                    result = tmp_output_text.split("assistant\n\n")[1]
                except IndexError:
                    result = tmp_output_text.split("assistant\n\n")[0]            
            else:
                result = tmp_output_text
            prediction_texts.append(result)

            # Save the prediction choice
            candidate_logits = []
            for label in ["A", "B", "C", "D", "E"]:
                try:
                    label_ids = tokenizer.encode(label, add_special_tokens=False)
                    label_id = label_ids[-1]  # Get the last token ID
                    candidate_logits.append(output[num].outputs[0].logprobs[0][label_id].logprob)
                except:
                    # print(f"Warning: {label} not found. Artificially adding log prob of -100.")
                    candidate_logits.append(-100)
            
            candidate_logits = torch.tensor(candidate_logits).to(torch.float32)
            prob = torch.nn.functional.softmax(candidate_logits, dim=0).detach().cpu().numpy()
            answer = {i: k for i, k in enumerate(["A", "B", "C", "D", "E"])}[np.argmax(prob)]
            predictions.append(answer)
            probs.append({'A': float(prob[0]), 'B': float(prob[1]), 'C': float(prob[2]), 'D': float(prob[3]), 'E': float(prob[4])})
        
        with open(output_file, 'a', encoding='utf-8') as f:
            for j in range(20000):
                if i + j >= len(all_datas):
                    break
                data = all_datas[i + j]
                data['prediction'] = predictions[j]
                data['prob'] = probs[j]
                data['prediction_text'] = prediction_texts[j]
                f.write(json.dumps(data) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference on an input JSONL file using a specified model.")
    parser.add_argument('--model_path', type=str, help="Path to the pre-trained model.")
    parser.add_argument('--input_file', type=str, help="Path to the input JSONL file.")
    parser.add_argument('--out_dir', type=str, help="Directory to save the output JSON file.")
    args = parser.parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    inference(args.model_path, args.input_file, args.out_dir)