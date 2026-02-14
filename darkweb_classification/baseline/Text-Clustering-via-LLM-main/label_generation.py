import random
from openai import OpenAI
import os
import json
import argparse
from datetime import datetime
from tqdm import tqdm
import time
import re

def ini_client(api_key: str):
    if not api_key or api_key.strip() == "sk-":
        raise ValueError("API key is missing or invalid!")
    client = OpenAI(
        api_key=api_key.strip(),
        base_url="https://api.chatanywhere.tech/v1"
    )
    return client

def chat(prompt, client):
    completion = client.chat.completions.create(
        model="gpt-5-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

def load_dataset(data_path, data, use_large):
    data_file = os.path.join(data_path, data, "large.jsonl") if use_large else os.path.join(data_path, data, "small.jsonl")
    print(f"Use dataset {data_file}")
    with open(data_file, 'r') as f:
        data_list = [json.loads(line) for line in f]
    print(f"Length of dataset: {len(data_list)}")
    return data_list

def get_label_list(data_list):
    return list({data["label"] for data in data_list})

def prompt_construct_generate_label(sentence_list, given_labels):
    json_example = {"labels": ["label name", "label name"]}
    return f"""Given the labels, under a text classification scenario, can all these text match the label given? If the sentence does not match any of the label, please generate a meaningful new label name.
Labels: {given_labels}
Sentences: {sentence_list}
You should NOT return meaningless label names such as 'new_label_1' or 'unknown_topic_1' and only return the new label names, please return in json format like: {json_example}"""

def prompt_construct_merge_label(label_list):
    json_example = {"merged_labels": ["label name", "label name"]}
    return f"""Please analyze the provided list of labels to identify entries that are similar or duplicate, considering synonyms, variations in phrasing, and closely related terms that essentially refer to the same concept. Your task is to merge these similar entries into a single representative label for each unique concept identified. The goal is to simplify the list by reducing redundancies without organizing it into subcategories or altering its fundamental structure.
Here is the list of labels for analysis and simplification: {label_list}.
Produce the final, simplified list in a flat, JSON-formatted structure without any substructures or hierarchical categorization like: {json_example}"""

def get_sentences(sentence_list):
    return [i['input'] for i in sentence_list]

def label_generation(args, client, data_list, chunk_size):
    with open(args.given_label_path, 'r') as f:
        given_labels = json.load(f)
    all_labels = given_labels[args.data].copy()
    
    for i in range(0, min(len(data_list), args.test_num * chunk_size), chunk_size):
        sentences = get_sentences(data_list[i:i+chunk_size])
        prompt = prompt_construct_generate_label(sentences, given_labels[args.data])
        try:
            response = chat(prompt, client)
            parsed = json.loads(response)
            new_labels = parsed.get("labels", [])
            for label in new_labels:
                if "unknown_topic" not in label and "new_label" not in label and label not in all_labels:
                    all_labels.append(label)
        except Exception as e:
            print(f"Error processing batch {i//chunk_size}: {e}")
            continue
    return all_labels

def merge_labels(args, all_labels, client): 
    prompt = prompt_construct_merge_label(all_labels)
    try:
        response = chat(prompt, client)
        parsed = json.loads(response)
        return parsed.get("merged_labels", all_labels)
    except:
        return all_labels

def write_dict_to_json(args, input, output_path, output_name):
    size = "large" if args.use_large else "small"
    file_name = os.path.join(output_path, f"{args.data}_{size}_{output_name}.json")
    with open(file_name, 'w') as json_file:
        json.dump(input, json_file, indent=2, ensure_ascii=False)
    print(f"JSON file '{file_name}' written.")

def main(args): 
    print("use_large: ", args.use_large)
    start_time = time.time()
    client = ini_client(args.api_key)  # ✅ 修复点：传入 api_key
    data_list = load_dataset(args.data_path, args.data, args.use_large)
    random.shuffle(data_list)
    label_list = get_label_list(data_list)
    print(f"Total cluster num: {len(label_list)}")
    write_dict_to_json(args, label_list, args.output_path, "true_labels")
    print(sorted(label_list))
    
    all_labels = label_generation(args, client, data_list, args.chunk_size)
    print(f"Total labels given by LLM: {len(all_labels)}")
    write_dict_to_json(args, all_labels, args.output_path, "llm_generated_labels_before_merge")
    
    final_labels = merge_labels(args, all_labels, client)
    write_dict_to_json(args, final_labels, args.output_path, "llm_generated_labels_after_merge")
    print(f"Label number after merge: {len(final_labels)}")
    print(final_labels)
    print(f"Total time usage: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./dataset/")
    parser.add_argument("--data", type=str, default="arxiv_fine")
    parser.add_argument("--output_path", type=str, default="./generated_labels")
    parser.add_argument("--given_label_path", type=str, default="./generated_labels/chosen_labels.json")
    parser.add_argument("--use_large", action="store_true")
    parser.add_argument("--print_details", type=bool, default=False)
    parser.add_argument("--test_num", type=int, default=5)
    parser.add_argument("--chunk_size", type=int, default=15)
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key")  # ✅ 改为 required
    args = parser.parse_args()
    main(args)