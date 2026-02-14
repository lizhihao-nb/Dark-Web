import json
import numpy as np
import re
import warnings
from bs4 import BeautifulSoup
import pymongo

# --- sklearn 部分 ---
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
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
collection = db["content20000"]  # ← 修改为你的目标集合

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
        continue  # 跳过异常文档

# 对齐长度
min_len = min(len(original_alerts), len(processed_texts))
original_alerts = original_alerts[:min_len]
processed_texts = processed_texts[:min_len]

print(f"✅ 成功加载并清洗 {len(original_alerts)} 条有效文本")

if not original_alerts:
    raise ValueError("没有有效文本可用于聚类！")

# ==================== 4. 聚类逻辑 ====================

def find_optimal_topics_lda(doc_term_matrix, max_topics=50):
    max_k = min(max_topics + 1, doc_term_matrix.shape[0])
    if max_k <= 2:
        return 2, []
    topic_range = range(5, max_k, 5)  # 从5开始，步长5，减少计算
    if 2 not in topic_range:
        topic_range = [2] + list(topic_range)
    topic_range = sorted(set(topic_range))
    
    perplexities = []
    print("🔍 寻找最优 LDA 主题数（基于困惑度）...")
    best_n, best_ppl = 2, float('inf')
    for n in topic_range:
        if n >= doc_term_matrix.shape[0]:
            break
        lda = LatentDirichletAllocation(n_components=n, random_state=42, max_iter=50)
        lda.fit(doc_term_matrix)
        ppl = lda.perplexity(doc_term_matrix)
        perplexities.append(ppl)
        print(f"  Topics: {n}, Perplexity: {ppl:.2f}")
        if ppl < best_ppl:
            best_ppl = ppl
            best_n = n
    return best_n, perplexities

def find_optimal_clusters_kmeans_fast(tfidf_matrix, max_clusters=50):
    """使用 inertia（肘部法）快速估计合理聚类数，避免 silhouette"""
    max_k = min(max_clusters + 1, tfidf_matrix.shape[0])
    if max_k <= 2:
        return 2
    # 对于大数据，默认选择一个合理值（如30），或根据经验调整
    # 这里我们简单返回 min(30, max_k-1)
    chosen = min(30, max_k - 1)
    print(f"💡 K-means 聚类数设为 {chosen}（大数据跳过 silhouette 计算）")
    return chosen

# ========== 准备向量表示 ==========
print("\n🧠 准备向量表示（max_features=1000）...")
vectorizer_lda = CountVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
doc_term_matrix_lda = vectorizer_lda.fit_transform(processed_texts)

vectorizer_kmeans = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
tfidf_matrix = vectorizer_kmeans.fit_transform(processed_texts)

# ========== 自动选择最优参数 ==========
optimal_lda, _ = find_optimal_topics_lda(doc_term_matrix_lda, max_topics=50)
optimal_kmeans = find_optimal_clusters_kmeans_fast(tfidf_matrix, max_clusters=50)

print(f"\n📌 最优 LDA 主题数: {optimal_lda}")
print(f"📌 K-means 聚类数: {optimal_kmeans}\n")

# ========== LDA ==========
print("🧩 运行 LDA 聚类...")
lda = LatentDirichletAllocation(n_components=optimal_lda, random_state=42, max_iter=100)
lda_doc_topic = lda.fit_transform(doc_term_matrix_lda)
lda_topics = np.argmax(lda_doc_topic, axis=1)

feature_names_lda = vectorizer_lda.get_feature_names()
lda_labels = {}
for i in range(optimal_lda):
    top_words = [feature_names_lda[idx] for idx in lda.components_[i].argsort()[-3:][::-1]]
    lda_labels[i] = '_'.join(top_words)
    print(f"LDA 主题 {i}: {lda_labels[i]}")

lda_results = [
    {"alert": text, "category": lda_labels[topic]}
    for text, topic in zip(original_alerts, lda_topics)
]

with open('lda_results.json', 'w', encoding='utf-8') as f:
    json.dump(lda_results, f, ensure_ascii=False, indent=2)
print(f"✅ 已保存 LDA 结果到 lda_results.json ({len(lda_results)} 条)")

# ========== K-means ==========
print("\n🧩 运行 K-means 聚类...")
kmeans = KMeans(n_clusters=optimal_kmeans, random_state=42, n_init=5, max_iter=100)
kmeans_labels = kmeans.fit_predict(tfidf_matrix)

feature_names_km = vectorizer_kmeans.get_feature_names()
kmeans_cluster_labels = {}
for i in range(optimal_kmeans):
    mask = (kmeans_labels == i)
    if np.sum(mask) == 0:
        label = "empty"
    else:
        avg_tfidf = np.array(tfidf_matrix[mask].mean(axis=0)).flatten()
        top_words = [feature_names_km[idx] for idx in avg_tfidf.argsort()[-3:][::-1]]
        label = '_'.join(top_words)
    kmeans_cluster_labels[i] = label
    print(f"K-means 聚类 {i}: {label}")

kmeans_results = [
    {"alert": text, "category": kmeans_cluster_labels[label]}
    for text, label in zip(original_alerts, kmeans_labels)
]

with open('tfidf_results.json', 'w', encoding='utf-8') as f:
    json.dump(kmeans_results, f, ensure_ascii=False, indent=2)
print(f"✅ 已保存 K-means 结果到 tfidf_results.json ({len(kmeans_results)} 条)")

# ========== 统计 ==========
from collections import Counter
print("\n" + "="*50)
print("📊 最终统计")
print("="*50)
print(f"LDA 使用 {optimal_lda} 个主题")
print(f"K-means 使用 {optimal_kmeans} 个聚类")

lda_dist = Counter(lda_topics)
print("\nLDA 主题分布:")
for tid, cnt in lda_dist.most_common():
    print(f"  {lda_labels[tid]}: {cnt} 条")

kmeans_dist = Counter(kmeans_labels)
print("\nK-means 聚类分布:")
for cid, cnt in kmeans_dist.most_common():
    print(f"  {kmeans_cluster_labels[cid]}: {cnt} 条")

print("\n🎉 全部完成！结果已保存为 *_20k.json 文件。")