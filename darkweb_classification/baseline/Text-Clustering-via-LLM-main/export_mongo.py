# export_mongo.py
import json, os, sys
import pymongo
from bs4 import BeautifulSoup
import re

def extract_clean_text(html):
    try:
        soup = BeautifulSoup(html, 'lxml')
        for t in soup(['script','style','meta','link','nav','footer','header']):
            t.decompose()
        text = soup.get_text(separator=' ', strip=True)
        return re.sub(r'\s+', ' ', text).strip()
    except:
        return re.sub(r'<[^>]+>', '', html)

with open("config.json") as f:
    cfg = json.load(f)

client = pymongo.MongoClient(cfg["mongo_uri"])
col = client[cfg["db_name"]][cfg["collection_name"]]
name = cfg["dataset_name"]
os.makedirs(f"./dataset/{name}", exist_ok=True)

with open(f"./dataset/{name}/small.jsonl", "w") as f:
    for doc in col.find({}, {"content": 1, "summary": 1}):
        cont = doc.get("content", "")
        summ = doc.get("summary", "").strip()
        if not cont or not summ or summ == "内容过短，无法生成摘要":
            continue
        clean = extract_clean_text(cont)
        if len(clean) < 20:
            continue
        f.write(json.dumps({
            "input": clean,
            "summary": summ,
            "label": "unknown"
        }, ensure_ascii=False) + "\n")
print("✅ MongoDB 导出完成")