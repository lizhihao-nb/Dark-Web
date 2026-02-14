#!/bin/bash
set -e

# 读取配置
DATASET_NAME=$(python3 -c "import json; print(json.load(open('config.json'))['dataset_name'])")
API_KEY=$(python3 -c "import json; print(json.load(open('config.json'))['openai_api_key'])")

# 创建目录
mkdir -p ./generated_labels ./logs/${DATASET_NAME}_small

# 写入 chosen_labels.json
python3 -c "
import json
with open('config.json') as f:
    cfg = json.load(f)
with open('./generated_labels/chosen_labels.json', 'w') as out:
    json.dump({cfg['dataset_name']: cfg['initial_categories']}, out, ensure_ascii=False, indent=2)
"

LOG_DIR="./logs/${DATASET_NAME}_small"

# Step 1: Label Generation（同时输出到屏幕和日志）
echo "🚀 Step 1: Label Generation"
python label_generation.py \
  --data "${DATASET_NAME}" \
  --given_label_path "./generated_labels/chosen_labels.json" \
  --output_path "./generated_labels" \
  --api_key "${API_KEY}" \
  --test_num 5 \
  --chunk_size 1000 \
  2>&1 | tee "${LOG_DIR}/label_generation.log"

# Step 2: Classification
echo "🚀 Step 2: Classification"
python given_label_classification.py \
  --data "${DATASET_NAME}" \
  --output_path "./generated_labels" \
  --output_file_name "find_labels.json" \
  --api_key "${API_KEY}" \
  --test_num 1000 \
  2>&1 | tee "${LOG_DIR}/given_label_classification.log"

echo "✅ 论文流程完成！日志保存在: ${LOG_DIR}"