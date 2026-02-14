import json
import random
import requests
from collections import defaultdict
import re
import time
import argparse

# --- 配置 ---
API_URL = "https://api.chatanywhere.tech/v1/chat/completions"
API_KEY = ""
MODEL_NAME = "gpt-5-mini"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}
random.seed(42)


WORD_INTRUSION_PROMPT = """You are an expert in dark web content analysis. Among the following 6 texts:
- 5 belong to the category: **"{positive_category}"**
- 1 belongs to a different category: **"{negative_category}"** (this is the "intrusion item")

To correctly identify the intrusion, use this functional understanding of common dark web categories:

- **Marketplaces**: Platforms for trading — includes directories, vendor shops, forums (e.g., Dread), access queues, security checkpoints, CAPTCHA pages, and archived snapshots of such platforms.
- **Financial Services**: Standalone money movement — e.g., fake Western Union, crypto mixing, "legit hack" transfers. These are payment enablers, not trading platforms.
- **Hacking/Cybercrime**: Tools, exploits, tutorials, or services for system compromise.
- **Stolen Data**: Listings of compromised credentials, documents, or personal info.
- **Illicit Goods**: Direct offers of drugs, weapons, counterfeit items, etc.


Carefully assess the **primary intent and functional role** of each text.  
First, provide a brief reasoning. Then, output only: "Answer: X" (X = 1–6).

Text list:
{sample_list}

Do not output anything else after the answer."""


# --- 全局结果存储 ---
evaluation_results = {
    'word_intrusion_accuracy': 0.0,
    'word_intrusion_adjusted': 0.0,
    'word_intrusion_details': [],
}


def call_llm(messages, system_prompt="You are an expert in dark web content analysis and classification.", temperature=0.0):
    time.sleep(3)  # 防止 API 限流
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature
    }
    if system_prompt:
        payload["messages"].insert(0, {"role": "system", "content": system_prompt})

    try:
        response = requests.post(API_URL, json=payload, headers=HEADERS, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    except requests.exceptions.Timeout:
        print("\n[ERROR] ⏱️  API 请求超时（超过 30 秒）")
        return ""
    except requests.exceptions.ConnectionError:
        print("\n[ERROR] 🌐 网络连接失败（检查网络或 API 地址）")
        return ""
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        error_text = e.response.text[:500]
        print(f"\n[ERROR] 📡 HTTP {status_code}: {e}")
        print(f"   Response: {error_text}")
        return ""
    except requests.exceptions.RequestException as e:
        print(f"\n[ERROR] 🛠️  请求异常: {e}")
        return ""
    except KeyError as e:
        print(f"\n[ERROR] 🔑 响应格式错误（缺少字段）: {e}")
        if 'response' in locals():
            print(f"   Raw response: {response.text[:500]}")
        return ""
    except Exception as e:
        print(f"\n[ERROR] ❓ 未知错误: {type(e).__name__}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Response: {e.response.text[:500]}")
        return ""


def is_valid_text(text):
    if not text or len(text.strip()) < 20:
        return False
    if re.match(r'^[0-9\s\W]+$', text.strip()):
        return False
    return True


def word_intrusion_evaluation(data, total_rounds_target=1000):
    """Word Intrusion 评估：总共执行 total_rounds_target 轮（每轮 6 条文本）"""
    category_to_items = defaultdict(list)
    for item in data:
        cat = item.get('category', '')
        txt = item.get('alert', '')
        if cat and txt and is_valid_text(txt):
            category_to_items[cat].append(item)

    valid_categories = {
        cat: items for cat, items in category_to_items.items()
        if len(items) >= 5
    }

    if len(valid_categories) < 2:
        print("⚠️ 有效类别数少于2，无法进行 Word Intrusion 评估")
        return 0.0

    category_list = list(valid_categories.keys())
    detected_count = 0
    completed_rounds = 0
    details = []
    round_index = 0

    print(f"\n🔍 计划执行 {total_rounds_target} 轮 WI 测试（总共）...\n")

    while completed_rounds < total_rounds_target:
        # 随机选一个正类别
        positive_category = random.choice(category_list)
        positive_items = valid_categories[positive_category]
        if len(positive_items) < 5:
            continue

        # 随机选一个不同的负类别
        negative_candidates = [c for c in category_list if c != positive_category]
        if not negative_candidates:
            continue
        negative_category = random.choice(negative_candidates)
        negative_items = valid_categories[negative_category]
        if not negative_items:
            continue

        # 采样 5 个正样本（分层采样）
        sorted_pos = sorted(positive_items, key=lambda x: len(x['alert']))
        step = max(1, len(sorted_pos) // 5)
        selected_positive = [sorted_pos[i * step] for i in range(5)]
        if len(selected_positive) < 5:
            selected_positive = random.sample(positive_items, min(5, len(positive_items)))

        # 采样 1 个负样本
        neg_item = random.choice(negative_items)

        all_samples = selected_positive + [neg_item]
        random.shuffle(all_samples)

        # 构造文本（清理空白）
        sample_texts = []
        for i, s in enumerate(all_samples):
            text = s.get('alert', '').strip()
            text = ' '.join(text.split())  # 合并多余空白
            sample_texts.append(f"{i+1}. {text}")
        sample_list_str = '\n'.join(sample_texts)

        # 调用 LLM
        prompt = WORD_INTRUSION_PROMPT.format(
            positive_category=positive_category,
            negative_category=negative_category,
            sample_list=sample_list_str
        )
        messages = [{"role": "user", "content": prompt}]
        response = call_llm(messages, temperature=0.0)

        try:
            shuffled_index = all_samples.index(neg_item)
            expected_answer = str(shuffled_index + 1)
        except ValueError:
            print(f"[Round {round_index + 1}] 负样本丢失，跳过")
            round_index += 1
            continue

        # 解析 LLM 响应
        response_text = response.strip() if response else ""
        actual_answer = ""
        answer_match = re.search(r'Answer:\s*(\d+)', response_text, re.IGNORECASE)
        if answer_match:
            actual_answer = answer_match.group(1)
        else:
            lines = [line.strip() for line in response_text.split('\n') if line.strip()]
            if lines and lines[-1].isdigit() and 1 <= int(lines[-1]) <= 6:
                actual_answer = lines[-1]

        # 跳过无效答案
        if not actual_answer or not actual_answer.isdigit() or not (1 <= int(actual_answer) <= 6):
            print(f"\n[Round {round_index + 1}] LLM 未返回有效答案，跳过本轮。")
            round_index += 1
            continue

        is_correct = (actual_answer == expected_answer)
        if is_correct:
            detected_count += 1
            status = "✅ 正确"
        else:
            status = "❌ 错误"

        # 控制台输出（可注释以加速）
        print(f"\n{'─' * 60}")
        print(f"🔹 Round {completed_rounds + 1} / {total_rounds_target}")
        print(f"   正类别: {positive_category} | 负类别: {negative_category}")
        print(f"   预期: {expected_answer} | LLM: '{actual_answer}' → {status}")

        # 保存详细结果
        detail = {
            'round': completed_rounds + 1,
            'category': positive_category,
            'negative_category': negative_category,
            'expected_answer': expected_answer,
            'actual_answer': actual_answer,
            'is_detected': is_correct,
            'llm_response': response_text,
            'llm_reasoning': response_text
        }
        details.append(detail)
        completed_rounds += 1
        round_index += 1

    raw_accuracy = detected_count / completed_rounds if completed_rounds > 0 else 0.0
    evaluation_results['word_intrusion_accuracy'] = raw_accuracy
    evaluation_results['word_intrusion_adjusted'] = raw_accuracy
    evaluation_results['word_intrusion_details'] = details

    print(f"\n{'═' * 60}")
    print(f"📊 总体结果: {detected_count} / {completed_rounds} 正确")
    print(f"🎯 Word Intrusion 准确率: {raw_accuracy:.4f} ({raw_accuracy * 100:.2f}%)")
    return raw_accuracy


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_results(output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 详细结果（含完整推理）已保存至: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Word Intrusion (WI) 评估 —— 总共 N 轮（默认 1000）")
    parser.add_argument("input_file", help="输入 JSON 文件路径（需含 'alert' 和 'category' 字段）")
    parser.add_argument("--output-file", default="wi_evaluation_1000.json", help="输出结果文件")
    parser.add_argument("--total-rounds", type=int, default=1000, help="总共执行的 WI 轮数（默认 1000）")
    args = parser.parse_args()

    print("📥 加载数据...")
    data = load_data(args.input_file)
    print(f"✅ 成功加载 {len(data)} 条数据")

    acc = word_intrusion_evaluation(data, total_rounds_target=args.total_rounds)

    save_results(args.output_file)


if __name__ == "__main__":
    main()