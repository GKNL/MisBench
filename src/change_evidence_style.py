import json
import argparse
import random
import torch
import os
import ray
from vllm import LLM, SamplingParams
from tqdm import tqdm
from transformers import AutoTokenizer

"""
Basedo on generated wikipedia (Objective Language) style evidence, generate evidence in different styles:
1. News Report
2. Technical Language
3. Confident Language
4. Science Reference
"""

def process_evidence(evidence):
    evidence_pieces = evidence.split(".")
    evidence = ".".join(evidence_pieces[:-1])
    evidence = evidence.strip() + "."
    return evidence

def generate_and_write(llm, sampling_params, data_batch, output_dir, index, num_batch, conflict_type, evidence_style):
    """
    Generate evidence using the language model and write the results to a file.
    
    :param llm: LLM, the language model
    :param sampling_params: SamplingParams, parameters for sampling
    :param data_batch: list, batch of data to process
    :param output_dir: str, directory to save the output files
    :param index: int, current index in processing
    :param num_batch: int, number of data samples per batch
    :param conflict_type: str, type of conflict (e.g., 'correct', 'fact_conflict', 'temporal_conflict', 'semantic_conflict')
    :param evidence_style: str, type of style to generate (e.g., 'News Report', 'Technical Language', 'Confident Language', 'Science Reference')
    """
    prompts = [item[conflict_type + "_style_prompt"] for item in data_batch]
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
        # output_final = process_evidence(result)  # Remove the last sentence after the last "." (not for social media)
        output_final = result.strip()  # for social media
        data_batch[i][conflict_type + "_evidence_" + evidence_style] = output_final
        del data_batch[i][conflict_type + "_style_prompt"]
        del data_batch[i][conflict_type + "_style_prompt_category"]

    output_dir = os.path.join(output_dir, evidence_style)
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, f"output_{conflict_type}_{(index // num_batch)}.jsonl")
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for data in data_batch:
            output_file.write(json.dumps(data) + "\n")

def main(model_name, file_path, output_dir, num_batch, conflict_type, evidence_style):
    """
    Main function to process the input file and generate evidence using the language model.
    
    :param model_name: str, path to the pre-trained model
    :param file_path: str, path to the input JSON file
    :param output_dir: str, directory to save the output files
    :param num_batch: int, number of data samples per batch
    :param conflict_type: str, type of conflict (e.g., 'correct', 'fact_conflict', 'temporal_conflict', 'semantic_conflict')
    """

    # Initialize the language model with specific configurations
    llm = LLM(model=model_name, dtype="float16", tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.9, disable_log_stats=True)
    sampling_params = SamplingParams(temperature=0, max_tokens=512)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Templates for generating evidence
    templates = {
        "news_report": "Based on the provided text, please rewrite this text into a news report. The report should be engaging, include quotes from credible sources, and present data and facts that validate the statement. Do not alter the underlying content of the text. Only alter the style of the text. Answer with nothing but the changed text. Only answer with the altered text.",

        "technical_language": "Based on the provided text, please rewrite this text into a more technical style by using technical language. The resulting text should maintain an objective and technical tone. Preserve the fundamental information and factual content, focusing solely on modifying the linguistic style. Answer with nothing but the changed text. Only answer with the altered text.",

        "confident_language": "Based on the provided text, please alter this text so that it is extremeley confident. Each sentence should be clear and unambiguous. Do not alter the underlying content of the text. Only alter the style of the text. Answer with nothing but the changed text. Only answer with the altered text.",
        
        "science_reference": "Based on the provided text, please alter this text so that it include references to scientific sources. Do not alter the underlying content of the text. Only alter the style of the text. Answer with nothing but the changed text. Only answer with the altered text and corresponding references.",

        "blog": "Based on the provided text, please rewrite this text into a casual yet informative blog post. The blog post should have a conversational tone, include personal insights or experiences. Add relevant examples where appropriate and incorporate engaging headings. Use a mix of short and medium-length paragraphs. Do not alter the underlying content of the text. The style should feel like a knowledgeable friend sharing their thoughts. Answer with nothing but the changed text. Only answer with the altered text.",

        "social_media": "Based on the provided text, please rewrite this text into engaging social media post. The post should be attention-grabbing, written in a casual tone and suitable for platforms like Twitter. Include relevant hashtags and emojis that capture attention. Add call-to-actions (CTAs) where appropriate and use language that encourages engagement and sharing. Do not alter the underlying content of the text. Answer with nothing but the changed text. Only answer with the altered text."
    }

    os.makedirs(output_dir, exist_ok=True)

    data_batch = []

    with open(file_path, 'r', encoding='utf-8') as jsonl_file:
        for index, line in enumerate(jsonl_file):
            # start from index k * num_batch
            if index < num_batch * 3:
                continue
            
            item = json.loads(line)
            template = templates[evidence_style]

            origin_evidence = item[f"{conflict_type}_evidence"]
            user_template = f"Origin Text: {origin_evidence}\n\nNew Text:\n"

            messages = [
                {"role": "system", "content": template},
                {"role": "user", "content": user_template}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # system_template = 'system\n\n{system_message}'
            # template = system_template.format(system_message=template)
            item[f"{conflict_type}_style_prompt"] = text
            item[f"{conflict_type}_style_prompt_category"] = evidence_style

            data_batch.append(item)

            if (index + 1) % num_batch == 0:
                print(f"Batch processed. Writing to output at index: {index}")
                generate_and_write(llm, sampling_params, data_batch, output_dir, index, num_batch, conflict_type, evidence_style)
                data_batch = []

        if data_batch:
            generate_and_write(llm, sampling_params, data_batch, output_dir, index, num_batch, conflict_type, evidence_style)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate evidence using a pre-trained language model.")
    parser.add_argument('--model_name', type=str, help="Path to the pre-trained model.")
    parser.add_argument('--file_path', type=str, help="Path to the input JSON file.")
    parser.add_argument('--output_dir', type=str, help="Directory to save the output files.")
    parser.add_argument('--num_batch', type=int, default=20000, help="Number of data samples per batch.")
    parser.add_argument('--conflict_type', type=str, default="fact_conflict", choices=['correct', 'fact_conflict', 'temporal_conflict', 'semantic_conflict'], help="Type of conflict to process.")
    parser.add_argument('--evidence_style', type=str, default="news_report", choices=['news_report', 'technical_language', 'confident_language', 'science_reference', 'blog', 'social_media'], help="Type of style to generate.")
    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

    main(args.model_name, args.file_path, args.output_dir, args.num_batch, args.conflict_type, args.evidence_style)
