import json
import os
import argparse
import time
from openai import OpenAI

def ini_client(api_key: str):
    if not api_key or api_key.strip() == "sk-":
        raise ValueError("API key is missing or invalid!")
    return OpenAI(
        api_key=api_key.strip(),
        base_url="https://api.chatanywhere.tech/v1"
    )

def load_dataset(data_path, data, use_large=False):
    file_path = os.path.join(data_path, data, "large.jsonl" if use_large else "small.jsonl")
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

def get_predict_labels(output_path, data):
    file_path = os.path.join(output_path, f"{data}_small_llm_generated_labels_after_merge.json")
    with open(file_path, "r") as f:
        return json.load(f)

def classify_text(text, labels, client):
    prompt = f"""Text: {text}
Categories: {', '.join(labels)}
Which category does this text belong to? Respond with ONLY the exact category name or 'Unsuccessful' if none match."""
    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        result = response.choices[0].message.content.strip().strip('"')
        return result if result in labels else "Unsuccessful"
    except:
        return "Unsuccessful"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./dataset/")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./generated_labels")
    parser.add_argument("--output_file_name", type=str, default="find_labels.json")
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--test_num", type=int, default=10000)
    args = parser.parse_args()

    client = ini_client(args.api_key)  # ✅ 修复点
    data_list = load_dataset(args.data_path, args.data)
    label_list = get_predict_labels(args.output_path, args.data)
    
    result_dict = {label: [] for label in label_list}
    result_dict["Unsuccessful"] = []

    for i, item in enumerate(data_list[:args.test_num]):
        text = item["input"]
        pred = classify_text(text, label_list, client)
        result_dict[pred].append(text)
        if i % 10 == 0:
            print(f"Processed {i+1}/{min(len(data_list), args.test_num)}")

    output_file = os.path.join(args.output_path, f"{args.data}_small_{args.output_file_name}")
    with open(output_file, "w") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)
    print(f"✅ Classification result saved to {output_file}")

if __name__ == "__main__":
    main()