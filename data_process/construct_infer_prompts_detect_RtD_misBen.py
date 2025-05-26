"""
Used to construct the prompts (detection mode) for the RtD inference stage.
......
"""
import json
import os
import random
from transformers import AutoTokenizer

random.seed(42)

def prompt_two_evidence(tokenizer, question, evidence1, evidence2):
    basic_prompt = "Given two passages, please pick out the one that contain misinformation. Answer with a choice from the following options. Only return with a specific option.\n\n"
    # random shuffle the order of the evidences
    correct_option = "A"
    if random.random() > 0.5:
        evidence1, evidence2 = evidence2, evidence1
        # Swap the options
        correct_option = "B"
        

    evidence_prompt = f"Passage 1: {evidence1}\n\n\nPassage 2: {evidence2}\n\n"
    question_prompt = f"Question: Which passage is a piece of misinformation?\nOptions:\n"
    question_prompt += f"A. Passage 1\n"
    question_prompt += f"B. Passage 2\n"
    question_prompt += "Your choice:"
    # llama, qwen
    messages = [
        {"role": "system", "content": basic_prompt},
        {"role": "user", "content": evidence_prompt + question_prompt}
    ]
    # gemma
    # messages = [
    #     # {"role": "system", "content": basic_prompt},
    #     {"role": "user", "content": basic_prompt + evidence_prompt + question_prompt}
    # ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    prompt = text
    return prompt, "obj_obj", correct_option



dataset = "misBen"  # 2WikiMultihopQA  misBen
LLM = "Qwen2.5-14B-instruct"  # the LLM that generate evidence   llama-3-8B-instruct  Qwen2.5-7B-instruct
data_dir = f"/data/miaopeng/workplace/misbench/bank_output/evidence/{LLM}/misBen_subset/reconstruct"
data_file_name = f"output_correct_0.jsonl"

# load model tokenizer
token_LLM = "Qwen2.5-14B-instruct"  # Qwen2.5-0.5B-instruct   llama-3-70B-instruct
model_name = f"/data/miaopeng/workplace/misbench/hugging_cache/{token_LLM}"
output_dir = f"/data/miaopeng/workplace/misbench/data_inference_prompts/{token_LLM}-RtD/{dataset}_subset"
tokenizer = AutoTokenizer.from_pretrained(model_name)

all_data = []
question2answer = {}
relation2object = {}
with open(f"{data_dir}/{data_file_name}", 'r') as f:
    for line in f:
        data = json.loads(line)
        all_data.append(data)
        question = data["question"]
        correct_answer = data["object"]
        question2answer[question] = correct_answer
        relation = data["relation"]
        if relation not in relation2object:
            relation2object[relation] = []
        relation2object[relation].append(correct_answer)

# Construct the prompts
prompts = []
pair2prompt_method = {
    "correct_mis": prompt_two_evidence,
    "correct_temporal": prompt_two_evidence,
    "correct_semantic": prompt_two_evidence,
}


pair = "correct_semantic"  #  default  correct_mis  correct_temporal  correct_semantic

for i, data in enumerate(all_data):
    question = data["question"]
    correct_evidence = data["correct_evidence"]
    fact_conflict_evidence = data["fact_conflict_evidence"]
    temporal_conflict_evidence = data["temporal_conflict_evidence"]
    semantic_conflict_evidence = data["semantic_conflict_evidence"]

    # tmp_prompt, eivdence_style = prompt_two_evidence(tokenizer, question, options_dict, correct_evidence, fact_conflict_evidence)
    tmp_prompt, eivdence_style, correct_option = pair2prompt_method[pair](tokenizer, question, correct_evidence, semantic_conflict_evidence)  # correct mis
    # tmp_prompt, eivdence_style = pair2prompt_method[pair](tokenizer, question, options_dict, fact_conflict_evidence, semantic_conflict_evidence)  # correct temporal semantic
    tmp_json = {
        "question": question,
        "prompt": tmp_prompt,
        "evidence_style": eivdence_style,
        "options": str(["Yes", "NO"]) if correct_option == "A" else str(["NO", "Yes"]),  # "options": str(["Passage 1", "Passage 2"]),
        "correct_option": correct_option,
        "replaced_option": "B" if correct_option == "A" else "A",
    }
    prompts.append(tmp_json)

# Save the prompts
output_dir = os.path.join(output_dir, pair)
os.makedirs(output_dir, exist_ok=True)
output_file = f"{output_dir}/{pair}_style_{prompts[0]['evidence_style']}_detect_reconstruct.jsonl"
with open(output_file, 'w') as f:
    for prompt in prompts:
        f.write(json.dumps(prompt) + '\n')
