"""
Used to construct the prompts (detection mode) for the inference stage.
......
"""
import json
import os
import random
from transformers import AutoTokenizer

random.seed(42)


def prompt_single_detect(tokenizer, question, options_dict, evidence1, evidence2=None):
    basic_prompt = "Given a “passage”, determine whether or not it is a piece of misinformation according to your own knowledge, Only choose and return with a specific option.\n"

    evidence_prompt = f"Passage: {evidence1}\n\n"
    question_prompt = f"Question: Is the given 'passage' a piece of misinformation?\nOptions:\n"
    for option, content in options_dict.items():
        question_prompt += f"{option}. {content}\n"
    question_prompt += "Your choice:"
    messages = [
        {"role": "system", "content": basic_prompt},
        {"role": "user", "content": evidence_prompt + question_prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    prompt = text
    return prompt, "obj"





dataset = "misBen"
LLM = "llama-3-70B-instruct"  # the LLM that generate evidence
data_dir = f"/data/miaopeng/workplace/misbench/bank_output/nli_filter/{LLM}/self_check"
data_file_name = f"all_{dataset}_evidences.jsonl"

# load model tokenizer
token_LLM = "Qwen2.5-0.5B-instruct"  # Qwen2.5-0.5B-instruct   llama-3-70B-instruct
model_name = f"/data/miaopeng/workplace/misbench/hugging_cache/{token_LLM}"
output_dir = f"/data/miaopeng/workplace/misbench/data_inference_prompts/{token_LLM}/{dataset}"
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
    # "default": prompt_default,
    "single_detect_mis": prompt_single_detect,
    "single_detect_temporal": prompt_single_detect,
    "single_detect_semantic": prompt_single_detect,
    # "correct_mis": prompt_two_evidence,
    # "correct_temporal": prompt_two_evidence,
    # "correct_semantic": prompt_two_evidence
}


pair = "single_detect_temporal"  #  default  correct_mis  correct_temporal  correct_semantic  single_detect_mis  single_detect_temporal  single_detect_semantic

for i, data in enumerate(all_data):
    question = data["question"]
    correct_evidence = data["correct_evidence"]
    fact_conflict_evidence = data["fact_conflict_evidence"]
    temporal_conflict_evidence = data["temporal_conflict_evidence"]
    semantic_conflict_evidence = data["semantic_conflict_evidence"]

    # prepare the options
    options = ["YES", "NO"]
    random.shuffle(options)
    options_dict = {chr(65 + i): option for i, option in enumerate(options)}

    to_options = ['A', 'B']
    correct_ind = options.index("YES")
    replaced_ind = options.index("NO")
    correct_option = to_options[correct_ind]
    false_option = to_options[replaced_ind]


    # tmp_prompt, eivdence_style = prompt_two_evidence(tokenizer, question, options_dict, correct_evidence, fact_conflict_evidence)
    # tmp_prompt, eivdence_style = pair2prompt_method[pair](tokenizer, question, options_dict, correct_evidence, fact_conflict_evidence)  # correct mis
    tmp_prompt, eivdence_style = pair2prompt_method[pair](tokenizer, question, options_dict, temporal_conflict_evidence, semantic_conflict_evidence)  # correct temporal semantic fact_conflict_evidence
    tmp_json = {
        "question": question,
        "prompt": tmp_prompt,
        "evidence_style": eivdence_style,
        "options": str(options),
        "correct_option": correct_option,
        "false_option": false_option,
    }
    prompts.append(tmp_json)

# Save the prompts
output_dir = os.path.join(output_dir, pair)
os.makedirs(output_dir, exist_ok=True)
output_file = f"{output_dir}/{pair}_style_{prompts[0]['evidence_style']}_detect.jsonl"
with open(output_file, 'w') as f:
    for prompt in prompts:
        f.write(json.dumps(prompt) + '\n')
