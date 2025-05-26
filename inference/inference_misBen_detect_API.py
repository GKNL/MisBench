import json
import argparse
import random
import os
from openai import OpenAI
from vllm import LLM, SamplingParams
from tqdm import tqdm
from time import sleep
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

llm_to_api = {
    "gpt4": "gpt-4o",
    "gpt": "gpt-3.5-turbo-0125", 
    "claude": "claude-3-5-haiku-20241022",  # claude-3-haiku-20240307  claude-3-5-haiku-20241022
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "deepseek": "deepseek-chat",
    "llama8b": "meta-llama/Llama-3-8b-chat-hf",
    "llama": "meta-llama/Llama-3-70b-chat-hf",
    "qwen7b": "qwen1.5-7b-chat",
    "qwen": "qwen1.5-72b-chat",
    "gemma": "gemma-7b-it",
}

def call_api(llm, data_batch, output_dir, index, num_batch, conflict_type):
    """
    Call the API to generate evidence.
    
    :param llm: str, name of the language model
    :param data_batch: list, batch of data to process
    :param output_dir: str, directory to save the output files
    :param index: int, current index in processing
    :param num_batch: int, number of data samples per batch
    :param conflict_type: str, type of conflict (e.g., 'correct', 'fact_conflict', 'temporal_conflict', 'semantic_conflict')
    """

    if 'gpt' in llm:
        client = OpenAI(
            base_url = "https://35.aigcbest.top/v1",
            api_key = 'YOUR_API_KEY' 
        )
    elif 'deepseek' in llm:
        client = OpenAI(
            base_url = "https://api.deepseek.com",
            api_key = 'sk-f3d7740f09994edbb3a498445d52c75b'
        )
    elif 'qwen' in llm:
        client = OpenAI(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key="YOUR_API_KEY",
        )
    else:
        client = OpenAI(
            base_url = "https://api.aimlapi.com/",
            api_key = 'YOUR_API_KEY'
        )
    
    prompts_system = [item[conflict_type + "_prompt_system"] for item in data_batch]
    prompts_user = [item[conflict_type + "_prompt_user"] for item in data_batch]

    response_dict = defaultdict(dict)
    error_knt = 0
    for i, (system_prompt, user_prompt) in enumerate(zip(prompts_system, prompts_user)):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=llm_to_api[llm],
                seed=42,
                temperature=0.0
            )
            response_dict[i] = chat_completion.choices[0].message.content
            print(llm, i, response_dict[i])
        except Exception as e:
            print('Call API failed! ', e)
            sleep(1)
            response_dict[i] = 'Error!'
            error_knt += 1
    
    for i, output in enumerate(response_dict):
        output_text = output
        if "assistant" in output_text:
            try:
                result = output_text.split("assistant\n\n")[1]
            except IndexError:
                result = output_text.split("assistant\n\n")[0]            
        else:
            result = output_text
        # output_final = result.split("")[0] if "" in result else result
        output_final = result
        data_batch[i]["prediction"] = output_final
        del data_batch[i][conflict_type + "_prompt_system"]
        del data_batch[i][conflict_type + "_prompt_user"]

    output_file_path = os.path.join(output_dir, f"output_{conflict_type}_{(index // num_batch)}.jsonl")
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for data in data_batch:
            output_file.write(json.dumps(data) + "\n")


def call_api_multi_thread(llm, data_batch, output_dir, index, num_batch, conflict_type):
    """
    Call the API to generate evidence.
    
    :param llm: str, name of the language model
    :param data_batch: list, batch of data to process
    :param output_dir: str, directory to save the output files
    :param index: int, current index in processing
    :param num_batch: int, number of data samples per batch
    :param conflict_type: str, type of conflict (e.g., 'correct', 'fact_conflict', 'temporal_conflict', 'semantic_conflict')
    """

    if 'gpt' in llm:
        client = OpenAI(
            base_url = "https://35.aigcbest.top/v1",
            api_key = 'YOUR_API_KEY'
        )
    elif 'deepseek' in llm:
        client = OpenAI(
            base_url = "https://api.deepseek.com",
            api_key = 'sk-f3d7740f09994edbb3a498445d52c75b'
        )
    elif 'qwen' in llm:
        client = OpenAI(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key="YOUR_API_KEY",
        )
    else:
        client = OpenAI(
            base_url = "https://aigcbest.top/v1",
            api_key = 'YOUR_API_KEY'
        )
    
    prompts_system = [item[conflict_type + "_prompt_system"] for item in data_batch]
    prompts_user = [item[conflict_type + "_prompt_user"] for item in data_batch]

    response_dict = defaultdict(dict)
    error_knt = 0

    def process_prompt(i, system_prompt, user_prompt):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=llm_to_api[llm],
                seed=42,
                temperature=0.0,
                max_tokens=20,
            )
            response = chat_completion.choices[0].message.content
            print(llm, i, response)
            return i, response
        except Exception as e:
            print(i, 'Call API failed! ', e)
            sleep(1)
            return i, 'Error!'

    # def process_prompt(i, system_prompt, user_prompt):  # For test
    #     # print(i, system_prompt, user_prompt)
    #     print(i)
    #     return i, 'user_prompt'

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_prompt, i, system_prompt, user_prompt) for i, (system_prompt, user_prompt) in enumerate(zip(prompts_system, prompts_user))]
        for future in as_completed(futures):
            i, response = future.result()
            response_dict[i] = response
            if response == 'Error!':
                error_knt += 1

    for i, output in response_dict.items():
        output_text = output
        if "assistant" in output_text:
            try:
                result = output_text.split("assistant\n\n")[1]
            except IndexError:
                result = output_text.split("assistant\n\n")[0]            
        else:
            result = output_text
        output_final = result
        data_batch[i]["prediction"] = output_final
        del data_batch[i][conflict_type + "_prompt_system"]
        del data_batch[i][conflict_type + "_prompt_user"]

    out_name = ""
    if conflict_type == "fact_conflict":
        out_name = "mis"
    elif conflict_type == "temporal_conflict":
        out_name = "temporal"
    elif conflict_type == "semantic_conflict":
        out_name = "semantic"
    output_file_path = os.path.join(output_dir, f"single_detect_{out_name}_style_obj_detect.jsonl")
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for data in data_batch:
            output_file.write(json.dumps(data) + "\n")


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
        data_batch[i]["prediction"] = output_final
        del data_batch[i][conflict_type + "_prompt"] 

    output_file_path = os.path.join(output_dir, f"single_detect_{conflict_type}_{(index // num_batch)}.jsonl")
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for data in data_batch:
            output_file.write(json.dumps(data) + "\n")

def main(file_path, output_dir, num_batch, conflict_type):
    """
    Main function to process the input file and generate evidence using the language model.
    
    :param file_path: str, path to the input JSON file
    :param output_dir: str, directory to save the output files
    :param num_batch: int, number of data samples per batch
    :param conflict_type: str, type of conflict (e.g., 'correct', 'fact_conflict', 'temporal_conflict', 'semantic_conflict')
    """

    llm = "gpt4"  # deepseek  claude
    print(f"detecting on dataset {file_path}")
    print(f"Processing {llm} model in detecting {conflict_type}...")

    output_dir = os.path.join(output_dir, llm)
    os.makedirs(output_dir, exist_ok=True)

    data_batch = []

    with open(file_path, 'r', encoding='utf-8') as jsonl_file:
        for index, line in enumerate(jsonl_file):
            item = json.loads(line)

            system_template = "Given a “passage”, determine whether or not it is a piece of misinformation according to your own knowledge, Only choose and return with a specific option.\n"


            if conflict_type == 'fact_conflict' and item.get("fact_conflict_evidence"):
                evidence = item["fact_conflict_evidence"]
            elif conflict_type == 'temporal_conflict' and item.get("temporal_conflict_evidence"):
                evidence = item["temporal_conflict_evidence"]
            elif conflict_type == 'semantic_conflict' and item.get("semantic_conflict_evidence"):
                evidence = item["semantic_conflict_evidence"]


            evidence_prompt = f"Passage: {evidence}\n\n"
            question_prompt = f"Question: Is the given 'passage' a piece of misinformation?\nOptions:\n"
            question_prompt += f"A. Yes\n"
            question_prompt += f"B. No\n"
            question_prompt += "Give your choice and do not explain."  # "Your choice:"

            user_template = evidence_prompt + question_prompt
            
            # system_template = 'system\n\n{system_message}'
            # template = system_template.format(system_message=template)
            item[f"{conflict_type}_prompt_system"] = system_template
            item[f"{conflict_type}_prompt_user"] = user_template
            item["correct_option"] = "A"

            data_batch.append(item)

            if (index + 1) % num_batch == 0:
                print(f"Batch processed. Writing to output at index: {index}")
                # generate_and_write(llm, sampling_params, data_batch, output_dir, index, num_batch, conflict_type)
                # call_api(llm, data_batch, output_dir, index, num_batch, conflict_type)
                call_api_multi_thread(llm, data_batch, output_dir, index, num_batch, conflict_type)
                data_batch = []

        if data_batch:
            # generate_and_write(llm, sampling_params, data_batch, output_dir, index, num_batch, conflict_type)
            # call_api(llm, data_batch, output_dir, index, num_batch, conflict_type)
            call_api_multi_thread(llm, data_batch, output_dir, index, num_batch, conflict_type)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate evidence using a pre-trained language model.")
    parser.add_argument('--file_path', type=str, help="Path to the input JSON file.")
    parser.add_argument('--output_dir', type=str, help="Directory to save the output files.")
    parser.add_argument('--num_batch', type=int, default=1200, help="Number of data samples per batch.")
    parser.add_argument('--conflict_type', type=str, default="semantic_conflict", choices=['fact_conflict', 'temporal_conflict', 'semantic_conflict'], help="Type of conflict to process.")
    args = parser.parse_args()

    main(args.file_path, args.output_dir, args.num_batch, args.conflict_type)
