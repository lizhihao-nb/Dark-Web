from pymongo import MongoClient
from datetime import datetime

mongo_host = "mongodb://192.168.31.9:27017/"
db_name = "3our_spider_db"
collection_name = "content"

client = MongoClient(mongo_host)
db = client[db_name]
collection = db[collection_name]

print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始统计 category 分布...")

# 可选：先估算总文档数（带 category 的）
total_with_category = collection.count_documents({"category": {"$exists": True, "$ne": None}})
print(f"  - 共有 {total_with_category} 条包含有效 category 的文档")

# 执行聚合
pipeline = [
    {"$match": {"category": {"$exists": True, "$ne": None}}},
    {"$group": {"_id": "$category", "count": {"$sum": 1}}}
]

print("  - 正在执行聚合查询...")
result = list(collection.aggregate(pipeline))

print(f"[{datetime.now().strftime('%H:%M:%S')}] 查询完成，共 {len(result)} 个类别：")
print("-" * 40)

# 按数量降序排序（可选）
result.sort(key=lambda x: x["count"], reverse=True)

for item in result:
    cat = item["_id"]
    cnt = item["count"]
    print(f"Category: {cat:<30} | Count: {cnt:>8}")