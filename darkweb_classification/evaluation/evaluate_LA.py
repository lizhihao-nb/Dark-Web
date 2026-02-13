

# 这是label accuracy的题目数量
LABEL_SAMPLE_SIZE = 1000
# 这是word intrusion，每个类别的题目数量
INTRUSION_ROUNDS = 5
# 这是保存结果在哪个文件
OUTPUT_FILE = 'lightML_evaluation_results.json'

import json
import argparse
import random
import requests
from collections import defaultdict, Counter
import re
from typing import List, Dict, Tuple, Optional
import time
import math

# --- 配置 ---
API_URL = "https://api.chatanywhere.tech/v1/chat/completions"
API_KEY = "sk-cmz5LsPuRvfGFw9jhMa5Q89hoDVUoQYNaugjbX3zDIRDtIn6"
MODEL_NAME = "gpt-5-mini"  # 或你实际可用的模型
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}
random.seed(42)
EXPECTED_MAX_CATEGORIES = 100  # 你认为合理的最大类别数，可调

# --- Prompt模板（已适配暗网场景）---
LABEL_ACCURACY_PROMPT = """You are analyzing dark web content. Please select the most appropriate category for the following text.

Text: {alert}

Available categories:
1. {positive_category}
2. {negative_category}

Please **ONLY answer with number 1 or 2**."""

WORD_INTRUSION_PROMPT = """You are analyzing dark web content. Five texts belong to the same category, and one is from a different category (the intrusion).
Identify the intrusion by its number (1-6).

Texts:
{sample_list}

Please **ONLY answer with a single number (1-6)**."""

# --- 评估结果存储 ---
evaluation_results = {
    # 'regex_accuracy': 0.0,
    'label_accuracy': 0.0,
    # 'label_accuracy_adjusted': 0.0,
    'label_accuracy_details': [],
    'word_intrusion_accuracy': 0.0,
    # 'word_intrusion_adjusted': 0.0,
    'word_intrusion_details': [],
    'actual_category_count': 0,
    # 'category_penalty_factor': 1.0
}

def call_llm(messages, system_prompt="You are an expert in dark web content analysis and classification.", temperature=0.1):
    """调用 LLM API 的通用函数（适配暗网任务）"""
    time.sleep(3)  # 防止 API 限流
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
    }
    if system_prompt:
        payload["messages"].insert(0, {"role": "system", "content": system_prompt})

    try:
        response = requests.post(API_URL, json=payload, headers=HEADERS)
        response.raise_for_status()
        result = response.json()
        print(f"LLM Response: {result}")  # 调试用
        return result['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print(f"[Retrying] Error calling LLM API: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.text}")
        return call_llm(messages)
    except (KeyError, IndexError) as e:
        print(f"[Retrying] Error parsing LLM response: {e}")
        print(f"Response content: {response.text if 'response' in locals() else 'No response'}")
        return call_llm(messages)

def load_data(file_path):
    """加载JSON数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# def regex_based_evaluation(data):
#     """基于正则的评估：判断category是否出现在text中（无视大小写）"""
#     correct_count = 0
#     total_count = len(data)
    
#     for item in data:
#         text = item.get('alert', '')  # 保留字段名为 'alert' 以兼容输入
#         category = item.get('category', '')
        
#         if not category:
#             continue
            
#         if re.search(re.escape(category), text, re.IGNORECASE):
#             correct_count += 1
#         else:
#             text_clean = re.sub(r'\s+', '', text.lower())
#             category_clean = re.sub(r'\s+', '', category.lower())
#             if category_clean in text_clean:
#                 correct_count += 1
    
#     accuracy = correct_count / total_count if total_count > 0 else 0
#     evaluation_results['regex_accuracy'] = accuracy
#     print(f"基于正则的准确率（暗网文本）: {accuracy:.4f} ({correct_count}/{total_count})")
#     return accuracy

def label_accuracy_evaluation(data, sample_size):
    """Label Accuracy评估：使用LLM判断暗网分类是否合理"""
    all_categories = list(set(item.get('category', '') for item in data if item.get('category')))
    actual_category_count = len(all_categories)
    if actual_category_count < 2:
        print("类别数少于2，无法进行Label Accuracy评估")
        return 0.0

    sample_data = random.sample(data, min(sample_size, len(data)))
    correct_count = 0
    total_processed = 0
    details = []

    for i, item in enumerate(sample_data):
        original_category = item.get('category', '')
        text = item.get('alert', '')
        if not original_category or not text:
            continue

        negative_categories = [cat for cat in all_categories if cat != original_category]
        if not negative_categories:
            continue
        negative_category = random.choice(negative_categories)

        prompt = LABEL_ACCURACY_PROMPT.format(
            alert=text,
            positive_category=original_category,
            negative_category=negative_category
        )
        messages = [{"role": "user", "content": prompt}]
        response = call_llm(messages)

        detail = {
            'index': i,
            'text_snippet': text[:100] + '...' if len(text) > 100 else text,
            'positive_category': original_category,
            'negative_category': negative_category,
            'llm_response': response,
            'is_correct': False
        }

        # TODO: 这里其实判断比较草率
        if response:
            response = response.strip()
            if '1' in response:
                correct_count += 1
                detail['is_correct'] = True
            elif '2' in response:
                detail['is_correct'] = False
            total_processed += 1

        details.append(detail)

    raw_accuracy = correct_count / total_processed if total_processed > 0 else 0
    # penalty_factor = max(1.0, math.log(actual_category_count) / math.log(EXPECTED_MAX_CATEGORIES))
    # adjusted_accuracy = raw_accuracy / penalty_factor

    evaluation_results['label_accuracy'] = raw_accuracy
    # evaluation_results['label_accuracy_adjusted'] = adjusted_accuracy
    evaluation_results['label_accuracy_details'] = details
    evaluation_results['actual_category_count'] = actual_category_count
    # evaluation_results['category_penalty_factor'] = penalty_factor

    print(f"Label Accuracy (原始): {raw_accuracy:.4f} ({correct_count}/{total_processed})")
    # print(f"Label Accuracy (调整后, {actual_category_count}类 → penalty={penalty_factor:.2f}): {adjusted_accuracy:.4f}")
    return raw_accuracy

def word_intrusion_evaluation(data, rounds_per_category=5):
    """Word Intrusion评估：检测暗网类别内部一致性"""
    category_to_items = defaultdict(list)
    for item in data:
        cat = item.get('category', '')
        txt = item.get('alert', '')
        if cat and txt:
            category_to_items[cat].append(item)

    valid_categories = {cat: items for cat, items in category_to_items.items() if len(items) >= 5}
    if len(valid_categories) < 2:
        print("有效类别数少于2，无法进行Word Intrusion评估")
        return 0.0

    detected_count = 0
    total_rounds = 0
    details = []

    for category, positive_items in valid_categories.items():
        other_categories = [c for c in valid_categories if c != category]
        for _ in range(rounds_per_category):
            if len(positive_items) < 5 or not other_categories:
                continue

            selected_positive = random.sample(positive_items, 5)
            neg_cat = random.choice(other_categories)
            neg_item = random.choice(valid_categories[neg_cat])

            all_samples = selected_positive + [neg_item]
            random.shuffle(all_samples)

            sample_texts = [f"{i+1}. {s.get('alert', '')}" for i, s in enumerate(all_samples)]
            sample_list_str = '\n'.join(sample_texts)
            prompt = WORD_INTRUSION_PROMPT.format(sample_list=sample_list_str)

            messages = [{"role": "user", "content": prompt}]
            response = call_llm(messages)

            detail = {
                'category': category,
                'negative_category': neg_cat,
                'llm_response': response,
                'is_detected': False
            }

            try:
                shuffled_index = all_samples.index(neg_item)
                expected_answer = str(shuffled_index + 1)
                detail['expected_answer'] = expected_answer
                detail['actual_answer'] = response.strip() if response else ""

                if response and expected_answer in response:
                    detected_count += 1
                    detail['is_detected'] = True
                total_rounds += 1
            except ValueError:
                detail['error'] = 'Negative sample not found in shuffled list'

            details.append(detail)

    actual_category_count = len(category_to_items)
    evaluation_results['actual_category_count'] = actual_category_count
    # # 复用或计算 penalty factor
    # if 'category_penalty_factor' not in evaluation_results:
    #     actual_category_count = len(category_to_items)
    #     penalty_factor = max(1.0, math.log(actual_category_count) / math.log(EXPECTED_MAX_CATEGORIES))
    #     evaluation_results['actual_category_count'] = actual_category_count
    #     evaluation_results['category_penalty_factor'] = penalty_factor
    # else:
    #     penalty_factor = evaluation_results['category_penalty_factor']

    raw_accuracy = detected_count / total_rounds if total_rounds > 0 else 0
    # adjusted_accuracy = raw_accuracy / penalty_factor

    evaluation_results['word_intrusion_accuracy'] = raw_accuracy
    # evaluation_results['word_intrusion_adjusted'] = adjusted_accuracy
    evaluation_results['word_intrusion_details'] = details

    print(f"Word Intrusion Accuracy (原始): {raw_accuracy:.4f} ({detected_count}/{total_rounds})")
    # print(f"Word Intrusion Accuracy (调整后, penalty={penalty_factor:.2f}): {adjusted_accuracy:.4f}")
    return raw_accuracy

def save_results(output_file):
    """保存评估结果到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
    print(f"暗网分类评估结果已保存到: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='评估暗网文本分类效果')
    parser.add_argument('input_file', help='输入的JSON文件路径（每条含 "alert" 文本和 "category" 标签）')
    parser.add_argument('--output-file', default=OUTPUT_FILE, help='输出结果文件路径')
    parser.add_argument('--label-sample-size', type=int, default=LABEL_SAMPLE_SIZE, help='Label Accuracy评估的样本数量')
    parser.add_argument('--intrusion-rounds', type=int, default=INTRUSION_ROUNDS, help='每个类别的Word Intrusion评估轮数')
    
    args = parser.parse_args()
    
    print("正在加载暗网文本数据...")
    data = load_data(args.input_file)
    print(f"共加载 {len(data)} 条暗网文本")

    # print("\n=== 基于正则的评估（Baseline）===")
    # regex_acc = regex_based_evaluation(data)
    
    print("\n=== Label Accuracy 评估（LLM 判断分类合理性）===")
    label_acc = label_accuracy_evaluation(data, args.label_sample_size)
    
    print("\n=== Word Intrusion 评估（LLM 检测类别内聚性）===")
    intrusion_acc = word_intrusion_evaluation(data, args.intrusion_rounds)
    
    print("\n=== 暗网分类评估汇总 ===")
    # print(f"正则准确率:       {regex_acc:.4f}")
    print(f"Label Accuracy:   {label_acc:.4f}")
    print(f"Word Intrusion:   {intrusion_acc:.4f}")
    
    save_results(args.output_file)

    # # 类别健康度提示
    # cat_count = evaluation_results.get('actual_category_count', 0)
    # penalty = evaluation_results.get('category_penalty_factor', 1.0)
    # print(f"\n📊 分类体系健康度提示:")
    # print(f"   总类别数: {cat_count}")
    # print(f"   惩罚因子: {penalty:.2f}")
    # if cat_count > 2 * EXPECTED_MAX_CATEGORIES:
    #     print(f"   ⚠️  警告：类别严重过细（>{2*EXPECTED_MAX_CATEGORIES}）！建议合并相似类。")
    # elif cat_count > EXPECTED_MAX_CATEGORIES:
    #     print(f"   ℹ️  提示：类别数偏多（>{EXPECTED_MAX_CATEGORIES}），可能影响泛化。")
    # else:
    #     print(f"   ✅ 类别数量在合理范围内。")

if __name__ == "__main__":
    main()