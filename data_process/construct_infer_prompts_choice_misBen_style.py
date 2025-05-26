"""
Used to construct the prompts (choice mode) for the inference stage.
Different mode:
1. Two evidences
2. multiple evidences
......
"""
import json
import os
import random
from transformers import AutoTokenizer

random.seed(42)


def prompt_default(tokenizer, question, options_dict, evidence1, evidence2):
    basic_prompt = "According to your own knowledge, please choose the best choice from the following options. Only return with a specific option.\n\n"
    question_prompt = f"Question: {question}\nOptions:\n"
    for option, content in options_dict.items():
        question_prompt += f"{option}. {content}\n"
    question_prompt += "Your choice:"
    # llama, qwen
    messages = [
        {"role": "system", "content": basic_prompt},
        {"role": "user", "content": question_prompt}
    ]
    # gemma
    # messages = [
    #     # {"role": "system", "content": basic_prompt},
    #     {"role": "user", "content": basic_prompt + question_prompt}
    # ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    prompt = text
    return prompt, "obj"


def prompt_one_evidence_style(tokenizer, question, options_dict, evidence, evidence2=None, style="obj"):
    basic_prompt = "According to the evidence provided, please choose the best choice from the following options. Only return with a specific option.\n\n"


    evidence_prompt = f"Evidence: {evidence}\n\n"
    question_prompt = f"Question: {question}\nOptions:\n"
    for option, content in options_dict.items():
        question_prompt += f"{option}. {content}\n"
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
    return prompt, style


def prompt_two_evidence_style(tokenizer, question, options_dict, evidence1, evidence2, style1="obj", style2="obj"):
    basic_prompt = "According to the evidences provided, please choose the best choice from the following options. Only return with a specific option.\n\n"
    # random shuffle the order of the evidences
    if random.random() > 0.5:
        evidence1, evidence2 = evidence2, evidence1

    evidence_prompt = f"Evidence 1: {evidence1}\n\nEvidence 2: {evidence2}\n\n"
    question_prompt = f"Question: {question}\nOptions:\n"
    for option, content in options_dict.items():
        question_prompt += f"{option}. {content}\n"
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
    return prompt, style1 + "_" + style2






dataset = "misBen"  # 2WikiMultihopQA  misBen
LLM = "llama-3-70B-instruct"  # the LLM that generate evidence
evi_type = "fact_conflict"  # correct   fact_conflict   temporal_conflict  semantic_conflict


style = "technical_language"  # blog  news_report  science_reference  confident_language  technical_language

data_dir = f"/data/miaopeng/workplace/misbench/bank_output/evidence_style/{LLM}/misBen/{evi_type}/{style}"
data_file_name = f"/all_{evi_type}_{style}_{dataset}.jsonl"


# load model tokenizer
token_LLM = "llama-3-70B-instruct"  # Qwen2.5-0.5B-instruct   llama-3-70B-instruct
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
    "default": prompt_default,
    "correct_mis": prompt_two_evidence_style,
    "correct_temporal": prompt_two_evidence_style,
    "correct_semantic": prompt_two_evidence_style,
    "fact": prompt_one_evidence_style,
    "temporal": prompt_one_evidence_style,
    "semantic": prompt_one_evidence_style,
}

pair = "fact"  #  default  correct_mis  correct_temporal  correct_semantic

for i, data in enumerate(all_data):
    question = data["question"]
    correct_evidence = data["correct_evidence"]
    fact_conflict_style_evidence = data[f"fact_conflict_evidence_{style}"]
    # select an irrelevant answer from other questions
    irr_object = random.sample(relation2object[data['relation']], 1)[0]  # Randomly select a replaced object that connected to the same relation
    cnt = 5000
    while (irr_object == data['object']) and cnt > 0:
        irr_object = random.sample(relation2object[data['relation']], 1)[0]
        cnt -= 1
    # prepare the options
    options = [data["object"], data["replaced_object"], irr_object, "Not Sure", "Not in the options"]
    random.shuffle(options)
    options_dict = {chr(65 + i): option for i, option in enumerate(options)}

    to_options = ['A', 'B', 'C', 'D', 'E']
    correct_ind = options.index(data["object"])
    replaced_ind = options.index(data['replaced_object'])
    irrelavant_ind = options.index(irr_object)
    correct_option = to_options[correct_ind]
    replaced_option = to_options[replaced_ind]
    irrelavant_option = to_options[irrelavant_ind]


    # tmp_prompt, eivdence_style = prompt_two_evidence(tokenizer, question, options_dict, correct_evidence, fact_conflict_evidence)
    # tmp_prompt, eivdence_style = pair2prompt_method[pair](tokenizer, question, options_dict, correct_evidence, semantic_conflict_evidence)  # correct mis
    tmp_prompt, eivdence_style = pair2prompt_method[pair](tokenizer, question, options_dict, fact_conflict_style_evidence, fact_conflict_style_evidence, style=style)  # correct temporal semantic
    tmp_json = {
        "question": question,
        "prompt": tmp_prompt,
        "evidence_style": eivdence_style,
        "options": str(options),
        "correct_option": correct_option,
        "replaced_option": replaced_option,
        "irrelavant_option": irrelavant_option,
        "not_sure_option": to_options[options.index("Not Sure")],
        "not_in_option_option": to_options[options.index("Not in the options")],
    }
    prompts.append(tmp_json)

# Save the prompts
output_dir = os.path.join(output_dir, pair)
os.makedirs(output_dir, exist_ok=True)
output_file = f"{output_dir}/{pair}_style_{prompts[0]['evidence_style']}_choice.jsonl"
with open(output_file, 'w') as f:
    for prompt in prompts:
        f.write(json.dumps(prompt) + '\n')
