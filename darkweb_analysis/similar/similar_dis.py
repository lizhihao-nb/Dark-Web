from pymongo import MongoClient
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import csv

# === 配置 ===
mongo_uri = "mongodb://192.168.31.9:27017"
db_name = "3our_spider_db"
collection_name = "content"

TARGET_CATEGORIES = [
    "Illicit Marketplaces",
    "Darknet Forums & Directories", 
    "Financial Fraud & Money Laundering",
    "Hacking & Malware",
    "Illicit Drug Trade",
    "Firearms Trafficking", 
    "Child Sexual Exploitation",
    "Anonymity & OPSEC Guidance",
    "Illicit Hosting Infrastructure",
    "Digital Archiving & Preservation",
    "Law & Political Reporting"
]

# 使用 0.05 间隔：0.00–0.05, 0.05–0.10, ..., 0.95–1.00
bins = np.arange(0.0, 1.001, 0.05)  # 21个边界 → 20个区间
bin_labels = [f"{bins[i]:.2f}–{bins[i+1]:.2f}" for i in range(len(bins)-1)]

client = MongoClient(mongo_uri)
db = client[db_name]
collection = db[collection_name]

results = []

for category in TARGET_CATEGORIES:
    print(f"\nProcessing category: {category}")
    
    # 获取所有有效 vector
    docs = collection.find(
        {"category": category, "vector": {"$exists": True, "$type": "array"}},
        {"vector": 1}
    )
    
    vectors = []
    for doc in docs:
        vec = doc.get("vector")
        if isinstance(vec, list) and len(vec) > 0:
            try:
                vec = [float(x) for x in vec]
                vectors.append(vec)
            except (TypeError, ValueError):
                continue

    n = len(vectors)
    if n < 2:
        print(f"  → Skipped (only {n} vectors)")
        continue

    pair_count = n * (n - 1) // 2
    print(f"  → Computing {pair_count:,} pairs for {n} vectors...")

    # 全量计算余弦相似度
    X = np.array(vectors, dtype=np.float32)
    sim_matrix = cosine_similarity(X)
    similarities = sim_matrix[np.triu_indices(n, k=1)]

    # 分桶统计（0.05 粒度）
    counts, _ = np.histogram(similarities, bins=bins)
    bin_dict = {label: int(count) for label, count in zip(bin_labels, counts)}

    result = {
        "category": category,
        "vector_count": n,
        "pair_count": pair_count,
        "distribution": bin_dict
    }
    results.append(result)

    # 打印简要分布（可选）
    print(f"  → Done. Top intervals:")
    sorted_items = sorted(bin_dict.items(), key=lambda x: x[1], reverse=True)[:3]
    for label, cnt in sorted_items:
        print(f"      {label}: {cnt} pairs")

# === 保存为 CSV ===
with open("similarity_distribution_0.05.csv", "w", newline="", encoding="utf-8") as f:
    fieldnames = ["category", "vector_count", "pair_count"] + bin_labels
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in results:
        row = {
            "category": r["category"],
            "vector_count": r["vector_count"],
            "pair_count": r["pair_count"],
            **r["distribution"]
        }
        writer.writerow(row)

print("\n✅ All done! Results saved to 'similarity_distribution_0.05.csv'")