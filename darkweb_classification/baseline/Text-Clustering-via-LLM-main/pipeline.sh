#!/bin/bash
set -e

echo "=== Step 1: 从 MongoDB 导出数据 ==="
python export_mongo.py

echo "=== Step 2: 运行 LLM 辅助聚类 pipeline ==="
chmod +x run_mydata.sh
./run_mydata.sh

echo "=== Step 3: 转换为 alert 格式 ==="
python postprocess.py

echo "🎉 全流程完成！结果在 ./output/ 目录下。"