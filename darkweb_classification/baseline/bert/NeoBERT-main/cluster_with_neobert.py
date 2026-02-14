# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import json
import random
from sklearn.cluster import KMeans
from openai import OpenAI
import pymongo
from bs4 import BeautifulSoup
import re
from transformers import AutoTokenizer, AutoModel

# 配置
MONGO_HOST = "mongodb://192.168.31.9:27017/"
DB_NAME = "3our_spider_db"
COLLECTION_NAME = "content20000"
OPENAI_API_YOUR_KEY = "sk-cmz5LsPuRvfGFw9jhMa5Q89hoDVUoQYNaugjbX3zDIRDtIn6"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", OPENAI_API_YOUR_KEY)
OPENAI_BASE_URL = "https://api.chatanywhere.tech/v1"  # 移除末尾空格
N_CLUSTERS = 16
SAMPLES_PER_CLUSTER = 10
OUTPUT_PATH = "output/neobert_clustered_robust.json"
MODEL_NAME = "chandar-lab/NeoBERT"

def extract_clean_text(html_content: str) -> str:
    try:
        soup = BeautifulSoup(html_content, 'lxml')
        for tag in soup(['script', 'style', 'meta', 'link', 'nav', 'footer', 'header']):
            tag.decompose()
        text = soup.get_text(separator=' ', strip=True)
        return re.sub(r'\s+', ' ', text).strip()
    except Exception:
        return re.sub(r'<[^>]+>', '', html_content)

def main():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 移除末尾空格

    # 1. 读取数据
    print("📥 读取 MongoDB 数据...")
    client = pymongo.MongoClient(MONGO_HOST)
    collection = client[DB_NAME][COLLECTION_NAME]

    documents = []
    for doc in collection.find({"content": {"$exists": True}}, no_cursor_timeout=True):
        plain_text = extract_clean_text(doc["content"])
        if len(plain_text) < 50:
            continue

        summary = doc.get("summary", "").strip()
        documents.append({
            "text": plain_text,
            "summary": summary
        })

    print(f"✅ 加载 {len(documents)} 条有效文本")
    if len(documents) < N_CLUSTERS:
        print(f"❌ 文本数 ({len(documents)}) 少于聚类数 ({N_CLUSTERS})，退出")
        return

    # 2. 加载 NeoBERT
    print("🔄 加载 NeoBERT...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.eval()

    # 3. 提取嵌入
    embeddings = []
    for doc in documents:
        inputs = tokenizer(doc["text"], return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state[0, 0].cpu().numpy())
    embeddings = np.vstack(embeddings)

    # 4. 聚类
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    # 5. 随机抽样（每簇最多 SAMPLES_PER_CLUSTER 条）
    cluster_groups = {}
    for i, label in enumerate(labels):
        if label not in cluster_groups:
            cluster_groups[label] = []
        cluster_groups[label].append(documents[i]["text"])

    cluster_samples = {}
    for label, texts in cluster_groups.items():
        n_sample = min(SAMPLES_PER_CLUSTER, len(texts))
        sampled_texts = random.sample(texts, n_sample)
        cluster_samples[label] = sampled_texts
        print(f"聚类 {label}: 共 {len(texts)} 条 → 抽样 {n_sample} 条")

    # 6. LLM 直接用原始文本生成类别标签（跳过摘要）
    llm_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    cluster_categories = {}

    for cluster_id, texts in cluster_samples.items():
        # 直接使用原始文本（截断到 300 字符），避免超长
        prompt_texts = "\n".join(f"- {t[:300]}" for t in texts)
        prompt = f"""You are a text analysis expert. The following texts belong to the same semantic cluster. Please generate a concise category label consisting of 2 to 5 words.

Rules:
- Output ONLY the label, no explanation
- Use a noun phrase
- Avoid generic terms like "issue", "content", or "other"

Texts:
{prompt_texts}"""

        try:
            resp = llm_client.chat.completions.create(
                model="gpt-5-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                # max_tokens=100
            )

            # 检查响应结构
            if not resp.choices or not resp.choices[0].message.content:
                print(f"⚠️ 聚类 {cluster_id}: LLM 返回空内容")
                label = ""
            else:
                label = resp.choices[0].message.content.strip()

            cluster_categories[cluster_id] = label if label else f"类别_{cluster_id}"
            print(f"✅ 聚类 {cluster_id} 标签: '{label}'")

        except Exception as e:
            print(f"❌ 聚类 {cluster_id} LLM 调用异常: {repr(e)}")
            import traceback
            traceback.print_exc()
            cluster_categories[cluster_id] = f"类别_{cluster_id}"

    # 7. 输出所有文档（使用数据库中的 summary 作为 alert，并过滤无效项）
    output_data = []
    for i, doc in enumerate(documents):
        summary = doc.get("summary", "").strip()
        if not summary or summary == "内容过短，无法生成摘要":
            continue
        category = cluster_categories.get(labels[i], "未分类")
        output_data.append({
            "alert": summary,
            "category": category
        })

    # 8. 保存
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 完成！结果保存至: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()