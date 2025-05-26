import json
import argparse
import random
import os
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm

def generate_and_write(llm, sampling_params, data_batch, output_dir, index, num_batch, conflict_type):
    """
    Generate evidence using the language model and write the results to a file.
    
    :param llm: LLM, the language model
    :param sampling_params: SamplingParams, parameters for sampling
    :param data_batch: list, batch of data to process
    :param output_dir: str, directory to save the output files
    :param index: int, current index in processing
    :param num_batch: int, number of data samples per batch
    :param conflict_type: str, type of conflict (e.g., 'correct', 'fact_conflict', 'temporal_conflict', 'semantic_conflict')
    """
    prompts = [item[conflict_type + "_prompt"] for item in data_batch]
    outputs = llm.generate(prompts, sampling_params)
    for i, output in enumerate(outputs):
        output_text = output.outputs[0].text
        if "assistant" in output_text:
            try:
                result = output_text.split("assistant\n\n")[1]
            except IndexError:
                result = output_text.split("assistant\n\n")[0]            
        else:
            result = output_text
        # output_final = result.split("")[0] if "" in result else result
        output_final = result
        data_batch[i][conflict_type + "_evidence"] = output_final
        del data_batch[i][conflict_type + "_prompt"] 

    output_file_path = os.path.join(output_dir, f"output_{conflict_type}_{(index // num_batch)}.jsonl")
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for data in data_batch:
            output_file.write(json.dumps(data) + "\n")

def main(model_name, file_path, output_dir, num_batch, conflict_type):
    """
    Main function to process the input file and generate evidence using the language model.
    
    :param model_name: str, path to the pre-trained model
    :param file_path: str, path to the input JSON file
    :param output_dir: str, directory to save the output files
    :param num_batch: int, number of data samples per batch
    :param conflict_type: str, type of conflict (e.g., 'correct', 'fact_conflict', 'temporal_conflict', 'semantic_conflict')
    """
    # Initialize the language model with specific configurations
    llm = LLM(model=model_name, dtype="float16", tensor_parallel_size=torch.cuda.device_count())
    sampling_params = SamplingParams(temperature=0, max_tokens=512)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Templates for generating evidence
    templates = {
        'correct': {
            "wikipedia": "Following the provided description, create a Wikipedia entry that systematically presents comprehensive evidence for the given claim. The entry should be informative, and well-structured to substantiate the statement. Only output the document content."
        },
        'fact_conflict': {
            "wikipedia": "Based on the provided description, construct a Wikipedia entry that outlines a series of events and studies that are fictional but support the given claim. Ensure the entry maintains the formal tone and structure of a real Wikipedia article. Only output the document content."
        },
        'temporal_conflict': {
            "wikipedia": "Based on the provided description and time span, construct a Wikipedia entry. Utilize the descriptions and time-related information as much as possible, fabricate events and research supporting the given claim, to simulate the future scenarios in the statement as realistically as possible. Only output the document content."
        },
        'semantic_conflict': {
            "wikipedia": "Based on the provided description, construct a Wikipedia entry that outlines a series of events and studies that are fictional but support the given claim. Ensure the entry maintains the formal tone and structure of a real Wikipedia article. Only output the document content."
        }
    }

    os.makedirs(output_dir, exist_ok=True)

    data_batch = []

    with open(file_path, 'r', encoding='utf-8') as jsonl_file:
        for index, line in enumerate(jsonl_file):

            item = json.loads(line)
            key = random.choice(list(templates[conflict_type].keys()))
            system_template = templates[conflict_type][key] + " "

            user_template = ""
            if conflict_type in ['correct', "fact_conflict", 'temporal_conflict'] and item.get("subject_additional_information"):
                user_template += f'Description for "{item["subject"]}": {item["subject_additional_information"]}. '

            if conflict_type == 'semantic_conflict' and item.get('semantic_description'):
                user_template += f'Description for "{item["subject"]}": {item["semantic_description"]}. '            

            object_ = item["replaced_object"] if conflict_type in ["fact_conflict", 'temporal_conflict', 'semantic_conflict'] else item["object"]

            if conflict_type == "correct" and item.get("object_additional_information"):
                user_template += f'Description for "{object_}": {item["object_additional_information"]}. '
            if conflict_type in ["fact_conflict", 'temporal_conflict', 'semantic_conflict'] and item.get("replaced_object_additional_information"):
                user_template += f'Description for "{object_}": {item["replaced_object_additional_information"]}. '

            # choose the claim type based on the conflict type
            if conflict_type == "correct":
                claim_type = "correct_pair"
            elif conflict_type == "temporal_conflict":
                claim_type = "temporal_conflict_pair"
            else:  # fact_conflict, semantic_conflict
                claim_type = "fact_conflict_pair"

            claim_prompt = f"""\nClaim: {item.get(claim_type)}
Evidence:"""
            
            user_template += claim_prompt
            messages = [
                {"role": "system", "content": system_template},
                {"role": "user", "content": user_template}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            item[f"{conflict_type}_prompt"] = text
            item[f"{conflict_type}_prompt_category"] = key

            data_batch.append(item)

            if (index + 1) % num_batch == 0:
                print(f"Batch processed. Writing to output at index: {index}")
                generate_and_write(llm, sampling_params, data_batch, output_dir, index, num_batch, conflict_type)
                data_batch = []

        if data_batch:
            generate_and_write(llm, sampling_params, data_batch, output_dir, index, num_batch, conflict_type)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate evidence using a pre-trained language model.")
    parser.add_argument('--model_name', type=str, help="Path to the pre-trained model.")
    parser.add_argument('--file_path', type=str, help="Path to the input JSON file.")
    parser.add_argument('--output_dir', type=str, help="/data/miaopeng/workplace/conflictbank/bank_output/evidence/correct")
    parser.add_argument('--num_batch', type=int, default=20000, help="Number of data samples per batch.")
    parser.add_argument('--conflict_type', type=str, default="correct", choices=['correct', 'fact_conflict', 'temporal_conflict', 'semantic_conflict'], help="Type of conflict to process.")
    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
    main(args.model_name, args.file_path, args.output_dir, args.num_batch, args.conflict_type)
