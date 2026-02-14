# postprocess.py
import json, os, sys

with open("config.json") as f:
    cfg = json.load(f)
name = cfg["dataset_name"]

# 加载分类结果
with open(f"./generated_labels/{name}_small_find_labels.json") as f:
    pred = json.load(f)

# 构建 {原文本: 类别}
text2cat = {}
for cat, texts in pred.items():
    for t in texts:
        text2cat[t] = cat

# 读取 small.jsonl 并生成 alert 格式
out = []
with open(f"./dataset/{name}/small.jsonl") as f:
    for line in f:
        doc = json.loads(line)
        cat = text2cat.get(doc["input"], "其他")
        if cat == "Unsuccessful":
            cat = "其他"
        out.append({"alert": doc["summary"], "category": cat})

os.makedirs("./output", exist_ok=True)
with open(f"./output/{name}_alert_format.json", "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print(f"✅ 最终输出: ./output/{name}_alert_format.json ({len(out)} 条)")