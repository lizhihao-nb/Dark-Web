import json
import numpy as np
import re
import warnings
from bs4 import BeautifulSoup
import pymongo

# --- sklearn 部分 ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')

# ==================== 1. HTML 文本提取 ====================
def extract_clean_text(html_content: str) -> str:
    """使用 BeautifulSoup 从 HTML 提取干净文本"""
    if not html_content or not isinstance(html_content, str):
        return ""
    try:
        soup = BeautifulSoup(html_content, 'lxml')
        for tag in soup(['script', 'style', 'meta', 'link', 'nav', 'footer', 'header', 'aside']):
            tag.decompose()
        text = soup.get_text(separator=' ', strip=True)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        return re.sub(r'<[^>]+>', ' ', html_content)

# ==================== 2. 英文文本预处理 ====================
def simple_preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
    return ' '.join(text.split())

# ==================== 3. 从 MongoDB 读取 content20000 全量数据 ====================
print("📤 正在从 MongoDB 读取 content20000 集合中的全部 HTML 内容...")
client = pymongo.MongoClient("mongodb://192.168.31.9:27017/")
db = client["3our_spider_db"]
collection = db["content20000"]

original_alerts = []
processed_texts = []

cursor = collection.find(
    {"content": {"$exists": True, "$type": "string", "$ne": ""}},
    {"content": 1}
)

count = 0
for doc in cursor:
    try:
        html = doc["content"]
        clean_text = extract_clean_text(html)
        
        if len(clean_text) < 20:
            continue
            
        original_alerts.append(clean_text)
        en_only = simple_preprocess(clean_text)
        processed_texts.append(en_only if en_only else clean_text.lower())
        
        count += 1
        if count % 1000 == 0:
            print(f"  已处理 {count} 条...")
    except Exception as e:
        continue

# 对齐长度（安全起见）
min_len = min(len(original_alerts), len(processed_texts))
original_alerts = original_alerts[:min_len]
processed_texts = processed_texts[:min_len]

print(f"✅ 成功加载并清洗 {len(original_alerts)} 条有效文本")

if not original_alerts:
    raise ValueError("没有有效文本可用于聚类！")

# ==================== 4. K-means 聚类逻辑 ====================

def find_optimal_clusters_kmeans_fast(tfidf_matrix, max_clusters=20):
    """快速设定聚类数（大数据跳过复杂评估）"""
    max_k = min(max_clusters + 1, tfidf_matrix.shape[0])
    if max_k <= 2:
        return 2
    chosen = min(20, max_k - 1)  # 默认合理值
    print(f"💡 K-means 聚类数设为 {chosen}")
    return chosen

# ========== 准备 TF-IDF 向量表示 ==========
print("\n🧠 构建 TF-IDF 向量（max_features=100）...")
vectorizer = TfidfVectorizer(max_features=100, stop_words='english', ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(processed_texts)

# ========== 自动选择聚类数 ==========
optimal_kmeans = find_optimal_clusters_kmeans_fast(tfidf_matrix, max_clusters=50)
print(f"\n📌 K-means 聚类数: {optimal_kmeans}\n")

# ========== 执行 K-means ==========
print("🧩 运行 K-means 聚类...")
kmeans = KMeans(n_clusters=optimal_kmeans, random_state=42, n_init=5, max_iter=50)
kmeans_labels = kmeans.fit_predict(tfidf_matrix)

# ========== 为每个聚类生成标签（基于 top-3 TF-IDF 词）==========
feature_names = vectorizer.get_feature_names()
kmeans_cluster_labels = {}
for i in range(optimal_kmeans):
    mask = (kmeans_labels == i)
    if np.sum(mask) == 0:
        label = "empty"
    else:
        avg_tfidf = np.array(tfidf_matrix[mask].mean(axis=0)).flatten()
        top_indices = avg_tfidf.argsort()[-3:][::-1]
        top_words = [feature_names[idx] for idx in top_indices]
        label = '_'.join(top_words)
    kmeans_cluster_labels[i] = label
    print(f"K-means 聚类 {i}: {label}")

# ========== 保存结果 ==========
kmeans_results = [
    {"alert": text, "category": kmeans_cluster_labels[label]}
    for text, label in zip(original_alerts, kmeans_labels)
]

with open('tfidf_results.json', 'w', encoding='utf-8') as f:
    json.dump(kmeans_results, f, ensure_ascii=False, indent=2)
print(f"✅ 已保存 K-means 结果到 tfidf_results.json ({len(kmeans_results)} 条)")

# ========== 统计分布 ==========
from collections import Counter
print("\n" + "="*50)
print("📊 K-means 聚类分布")
print("="*50)
kmeans_dist = Counter(kmeans_labels)
for cid, cnt in kmeans_dist.most_common():
    print(f"  {kmeans_cluster_labels[cid]}: {cnt} 条")

print("\n🎉 K-means 聚类完成！结果已保存。")