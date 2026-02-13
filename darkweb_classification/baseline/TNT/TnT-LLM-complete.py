import requests
import json
import time
import random
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import lightgbm as lgb
import pickle
import os

import pymongo
import re

API_URL = "https://api.chatanywhere.tech/v1/chat/completions"
API_KEY = "sk-cmz5LsPuRvfGFw9jhMa5Q89hoDVUoQYNaugjbX3zDIRDtIn6"
MODEL_NAME = "gpt-5-mini"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# --- Prompt 模板 ---

prompt2 = """
# Instruction
## Context
- **Goal**: Your goal is to cluster the input data into meaningful categories for the given use case.
- **Data**: The input data will be a markdown table with summaries for a list of human-AI conversations, including the following columns:
    - **id**: conversation index.
    - **text**: conversation summary.
- **Use case**: {use_case}

## Requirements
### Format
- Output clusters as a **markdown table** with each row as a category, with the
following columns:
    - **id**: category index starting from 1 in an incremental manner.
    - **name**: category name should be **within {cluster_name_length} words**.It can be either *verb phrase* or *noun phrase*, whichever is more appropriate.
    - **description**: category description should be **within {cluster_description_length} words**.
Here is an example of your output:
```markdown
|id|name|description|
|-|-|-|
|category id|category name|category description|
```
- Total number of categories should be **no more than {max_num_clusters}**.
- Output table should be in **English** only.
### Quality
- **No overlap or contradiction** among the categories.
- **Name** is a concise and clear label for the category. Use only phrases that are specific to each category and avoid those that are common to all categories.
- **Description** differentiates one category from another.-**Name** and **description** can **accurately** and **consistently** classify new data points **without ambiguity**.
- **Name** and **description** are *consistent with each other*.
- Output clusters match the data as closely as possible, without missing important categories or adding unnecessary ones.
- Output clusters serve the given use case well.
- Output clusters should be specific and meaningful. Do not invent categories that are not in the data.

# Data
{data_table}
# Questions
## Q1. Please generate a cluster table from the input data that meets the requirements.
Tips
- The cluster table should be a **flat list** of **mutually exclusive** categories. Sort them based on their semantic relatedness.
- You can have *fewer than {max_num_clusters} categories* in the cluster table, but **do not exceed the limit.**
- Be **specific** about each category. **Do not include vague categories** such as, "General", "Unclear", "Miscellaneous" or "Undefined" in the cluster table.- You can ignore low quality or ambiguous data points.

## Q2. Why did you cluster the data the way you did? Explain your reasoning ** within {explanation_length} words**.

## Provide your answers between the tags: <cluster_table>your generated cluster table with no more than {max_num_clusters} categories</cluster_table>, <explanation>explanation of your reasoning process within {explanation_length} words</explanation>.

# Output
"""

prompt3 = """
# Instruction
## Context
- **Goal**: You goal is to review the given reference table based on the input data for the specified use case, then update the reference table if needed.
    - You will be given a reference cluster table, which is built on existing data. The reference table will be used to classify new data points.
    - You will compare the input data with the reference table, output a rating score of the quality of the reference table, suggest potential edits, and update the reference table if needed.
- **Reference cluster table**: The input cluster table is a markdown table with each row as a category, with the following columns:
    - **id**: category index.
    - **name**: category name.
    - **description**: category description used to classify data points.
- **Data**: The input data will be a markdown table with summaries for a list of human-AI conversations, including the following columns:
    - **id**: conversation index.
    - **text**: conversation summary.
- **Use case**: {use_case}

## Requirements
### Format
- Output clusters as a **markdown table** with each row as a category, with the
following columns:
    - **id**: category index starting from 1 in an incremental manner.
    - **name**: category name should be **within {cluster_name_length} words**.It can be either *verb phrase* or *noun phrase*, whichever is more appropriate.
    - **description**: category description should be **within {cluster_description_length} words**.
Here is an example of your output:
```markdown
|id|name|description|
|-|-|-|
|category id|category name|category description|
```
- Total number of categories should be **no more than {max_num_clusters}**.
- Output table should be in **English** only.
### Quality
- **No overlap or contradiction** among the categories.
- **Name** is a concise and clear label for the category. Use only phrases that are specific to each category and avoid those that are common to all categories.
- **Description** differentiates one category from another.-**Name** and **description** can **accurately** and **consistently** classify new data points **without ambiguity**.
- **Name** and **description** are *consistent with each other*.
- Output clusters match the data as closely as possible, without missing important categories or adding unnecessary ones.
- Output clusters serve the given use case well.
- Output clusters should be specific and meaningful. Do not invent categories that are not in the data.

# Reference cluster table
{cluster_table}

# Data
{data_table}

# Questions
## Q1: Review the given reference table and the input data and provide a rating score of the reference table. The rating score should be an integer between 0 and 100, higher rating score means better quality. You should consider the following factors when rating the reference cluster table:
- **Intrinstic quality**:
    - 1) if the cluster table meets the *Requirements* section, with clear and consistent category names and descriptions, and no overlap or contradiction among the categories;
    - 2) if the categories in the cluster table are relevant to the the given use case;
    - 3) if the cluster table includes any vague categories such as "General", "Unclear", "Miscellaneous" or "Undefined".
- **Extrinstic quality**:
    - 1) if the cluster table can accurately and consistently classify the input data without ambiguity;
    - 2) if there are missing categories in the cluster table but appear in theinput data;
    - 3) if there are unnecessary categories in the cluster table that do not appear in the input data.

## Q2: Explain your rating score in Q1 **within {explanation_length} words**.

## Q3: Based on your review, decide if you need to edit the reference table to improve its quality. If yes, suggest potential edits **within {suggestion_length} words**. If no, please output "N/A".
Tips:
- You can edit the category name, description, or remove a category. You can also merge or add new categories if needed. Your edits should meet the *Requirements* section.
- The cluster table should be a **flat list** of **mutually exclusive** categories. Sort them based on their semantic relatedness.
- You can have *fewer than {max_num_clusters} categories* in the cluster table, but **do not exceed the limit.**
- Be **specific** about each category. **Do not include vague categories** such as "General", "Unclear", "Miscellaneous" or "Undefined" in the cluster table.
- You can ignore low quality or ambiguous data points.

## Q4: If you decide to edit the reference table, please provide your updated reference table. If you decide not to edit the reference table, please output the original reference table.

## Provide your answers between the tags: <rating>your answer to Q1 between 0 and 100</rating>, <explanation>your answer to Q2 within {explanation_length} words</explanation>, <suggestion>your answer to Q3 within {suggestion_length} words</suggestion>, <cluster_table>your answer to Q4 in markdown table format with no more than {max_num_clusters} categories</cluster_table>.

# Output
"""

prompt4 = """
# Instruction
## Context
- **Goal**: You goal is to review the given reference table based on the requirements and the specified use case, then update the reference table if needed.
    - You will be given a reference cluster table, which is built on existing data. The reference table will be used to classify new data points.
    - You will compare the reference table with the given requirements, output a rating score of the quality of the reference table, suggest potential edits, and update the reference table if needed.
- **Reference cluster table**: The input cluster table is a markdown table with each row as a category, with the following columns:
    - **id**: category index.
    - **name**: category name.
    - **description**: category description used to classify data points.
- **Use case**: {use_case}

## Requirements
### Format
- Output clusters as a **markdown table** with each row as a category, with the
following columns:
    - **id**: category index starting from 1 in an incremental manner.
    - **name**: category name should be **within {cluster_name_length} words**.It can be either *verb phrase* or *noun phrase*, whichever is more appropriate.
    - **description**: category description should be **within {cluster_description_length} words**.
Here is an example of your output:
```markdown
|id|name|description|
|-|-|-|
|category id|category name|category description|
```
- Total number of categories should be **no more than {max_num_clusters}**.
- Output table should be in **English** only.
### Quality
- **No overlap or contradiction** among the categories.
- **Name** is a concise and clear label for the category. Use only phrases that are specific to each category and avoid those that are common to all categories.
- **Description** differentiates one category from another.-**Name** and **description** can **accurately** and **consistently** classify new data points **without ambiguity**.
- **Name** and **description** are *consistent with each other*.
- Output clusters match the data as closely as possible, without missing important categories or adding unnecessary ones.
- Output clusters serve the given use case well.
- Output clusters should be specific and meaningful. Do not invent categories that are not in the data.

# Reference cluster table
{cluster_table}

# Questions
## Q1: Review the given reference table and provide a rating score. The rating score should be an integer between 0 and 100, higher rating score means better quality. You should consider the following factors when rating the reference cluster table:
- **Intrinstic quality**:
    - 1) if the cluster table meets the *Requirements* section, with clear and consistent category names and descriptions, and no overlap or contradiction among the categories;
    - 2) if the categories in the cluster table are relevant to the the given use case;
    - 3) if the cluster table includes any vague categories such as "General", "Unclear", "Miscellaneous" or "Undefined".

## Q2: Explain your rating score in Q1 **within {explanation_length} words**.

## Q3: Based on your review, decide if you need to edit the reference table to improve its quality. If yes, suggest potential edits **within {suggestion_length} words**. If no, please output "N/A".
Tips:
- You can edit the category name, description, or remove a category. You can also merge or add new categories if needed. Your edits should meet the *Requirements* section.
- The cluster table should be a **flat list** of **mutually exclusive** categories. Sort them based on their semantic relatedness.
- You can have *fewer than {max_num_clusters} categories* in the cluster table, but **do not exceed the limit.**
- Be **specific** about each category. **Do not include vague categories** such as "General", "Unclear", "Miscellaneous" or "Undefined" in the cluster table.
- You can ignore low quality or ambiguous data points.

## Q4: If you decide to edit the reference table, please provide your updated reference table. If you decide not to edit the reference table, please output the original reference table.

## Provide your answers between the tags: <rating>your answer to Q1 between 0 and 100</rating>, <explanation>your answer to Q2 within {explanation_length} words</explanation>, <suggestion>your answer to Q3 within {suggestion_length} words</suggestion>, <cluster_table>your answer to Q4 in markdown table format with no more than {max_num_clusters} categories</cluster_table>.

# Output
"""

prompt5 = """
# Instruction
## Context
- **Goal**: Your goal is to classify the input data using the provided reference table.
- **Reference table**: The input reference table is a markdown table with each row as a category, with the following columns:
    - **id**: category index.
    - **name**: category name.
    - **description**: category description used to classify data points.
- **Data**: Your input data is a conversation history between a User and an AI agent.

# Reference table
{cluster_table}

# Data
{input_text}

# Questions
## Please classify the input data using the reference table. Your output should include the following information:
- **category-id**: **id** of a category in the reference table; if unable to classify using the reference table, please output "-1".
- **category-name**: **name** of a category in the reference table that corresponds to the **category-id**; if unable to classify using the reference table, please output "Undefined".
- **explanation**: a short explanation of why you think the input data belongs to the category or you cannot classify the data into any of the given categories. You explanation should be within {explanation_length} words.
Tips
- You should only output the **primary** category for the input data. If it can be classified into multiple categories, please output **the most relevant category**.
- Your output should be in *English* only.

## Please provide your answers between the tags: <category-id>your identified category id</category-id>, <category-name>your identified category name</category-name>, <explanation>your explanation</explanation>.

# Output
"""

# --- 辅助函数 ---
def call_llm(prompt, system_prompt="You are an expert in darkweb content analysis.", model=MODEL_NAME, max_tokens=4096, temperature=0.2):
    """调用 LLM API"""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
    }
    try:
        for _ in range(3):
            response = requests.post(API_URL, json=payload, headers=HEADERS)
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                print(f"API Error ({response.status_code}): {response.text}")
                time.sleep(2 ** _)
        return None
    except Exception as e:
        print(f"Request failed: {e}")
        return None

def load_data_from_mongodb(limit=20000):
    """从MongoDB加载数据，直接使用summary字段"""
    print("📤 正在从 MongoDB 读取数据...")
    client = pymongo.MongoClient("mongodb://192.168.31.9:27017/")
    db = client["3our_spider_db"]
    collection = db["content20000"]

    data = []
    cursor = collection.find(
        {"summary": {"$exists": True, "$ne": ""}},
        {"summary": 1, "vector": 1, "content": 1}
    ).limit(limit)

    for i, doc in enumerate(cursor):
        summary = doc.get("summary", "").strip()
        vector = doc.get('vector')
        content = doc.get("content", "").strip()
        
        if isinstance(vector, list):
            vector = np.array(vector).flatten()
        
        if summary and vector is not None and len(vector) > 0:
            data.append({
                'summary': summary,
                'vector': vector,
                'content': content[:200] if content else ""  # 保存前200字符用于调试
            })

    print(f"✅ 成功加载 {len(data)} 条数据")
    return data

# --- TnT-LLM Phase 1: Taxonomy Generation (使用summary字段) ---
def phase1_taxonomy_generation(summaries_sample):
    """
    执行 TnT-LLM 的第一阶段：生成标签分类法。
    使用已有的summary字段，跳过摘要生成。
    """
    print("--- Starting Phase 1: Taxonomy Generation ---")
    
    # --- Stage 1: 准备summary数据 ---
    print("Stage 1: Using existing summaries...")
    # 直接使用summary字段，不需要再生成摘要
    summary_items = []
    for i, summary in enumerate(summaries_sample):
        if summary:  # 确保summary非空
            summary_items.append({"id": i+1, "text": summary})
        else:
            # 如果summary为空，使用简短占位符
            summary_items.append({"id": i+1, "text": "Dark web content"})
    
    print(f"Using {len(summary_items)} summaries for taxonomy generation")
    
    # --- Stage 2: Taxonomy Creation, Update, and Review ---
    print("Stage 2: Creating and refining taxonomy...")
    
    # 定义分类法生成参数
    use_case = "Understand and categorize different types of dark web content based on their textual descriptions."
    cluster_name_length = 4
    cluster_description_length = 20
    max_num_clusters = 15
    explanation_length = 100
    suggestion_length = 100
    
    # 将数据分成小批次
    batch_size = 1000
    summary_batches = [summary_items[i:i + batch_size] for i in range(0, len(summary_items), batch_size)]
    
    final_taxonomy = None

    # Initial Generation (Prompt 2)
    if summary_batches:
        print("  Initial Taxonomy Generation...")
        first_batch = summary_batches[0]
        
        # 构建Markdown表格
        first_batch_table_lines = ["|id|text|", "|-|-|"]
        for item in first_batch:
            escaped_text = item['text'].replace("|", "\\|")
            first_batch_table_lines.append(f"|{item['id']}|{escaped_text}|")
        first_batch_table = "\n".join(first_batch_table_lines)
        
        p2_content = prompt2.format(
            use_case=use_case,
            data_table=first_batch_table,
            cluster_name_length=cluster_name_length,
            cluster_description_length=cluster_description_length,
            max_num_clusters=max_num_clusters,
            explanation_length=explanation_length
        )
        
        initial_tax_response = call_llm(p2_content)
        if initial_tax_response:
            try:
                start_tag = "<cluster_table>"
                end_tag = "</cluster_table>"
                start_idx = initial_tax_response.find(start_tag) + len(start_tag)
                end_idx = initial_tax_response.find(end_tag)
                cluster_table_text = initial_tax_response[start_idx:end_idx].strip()
                final_taxonomy = cluster_table_text
                print("  Initial taxonomy created.")
            except Exception as e:
                print(f"  Error parsing initial taxonomy: {e}")
                return None
        else:
            print("  Failed to create initial taxonomy.")
            return None
        time.sleep(1)

    # Iterative Update (Prompt 3)
    for i in range(1, len(summary_batches)):
        print(f"  Updating Taxonomy with batch {i+1}/{len(summary_batches)}...")
        current_batch = summary_batches[i]
        current_batch_table_lines = ["|id|text|", "|-|-|"]
        for item in current_batch:
            escaped_text = item['text'].replace("|", "\\|")
            current_batch_table_lines.append(f"|{item['id']}|{escaped_text}|")
        current_batch_table = "\n".join(current_batch_table_lines)

        p3_content = prompt3.format(
            use_case=use_case,
            cluster_table=final_taxonomy,
            data_table=current_batch_table,
            cluster_name_length=cluster_name_length,
            cluster_description_length=cluster_description_length,
            max_num_clusters=max_num_clusters,
            explanation_length=explanation_length,
            suggestion_length=suggestion_length
        )
        
        update_tax_response = call_llm(p3_content)
        if update_tax_response:
            try:
                start_tag = "<cluster_table>"
                end_tag = "</cluster_table>"
                start_idx = update_tax_response.find(start_tag) + len(start_tag)
                end_idx = update_tax_response.find(end_tag)
                updated_cluster_table_text = update_tax_response[start_idx:end_idx].strip()
                final_taxonomy = updated_cluster_table_text
                print(f"  Taxonomy updated with batch {i+1}.")
            except Exception as e:
                print(f"  Error parsing updated taxonomy from batch {i+1}: {e}. Keeping previous version.")
        else:
            print(f"  Failed to update taxonomy with batch {i+1}. Keeping previous version.")
        time.sleep(1)

    # Final Review (Prompt 4)
    print("  Final Review of Taxonomy...")
    p4_content = prompt4.format(
        use_case=use_case,
        cluster_table=final_taxonomy,
        cluster_name_length=cluster_name_length,
        cluster_description_length=cluster_description_length,
        max_num_clusters=max_num_clusters,
        explanation_length=explanation_length,
        suggestion_length=suggestion_length
    )
    
    review_tax_response = call_llm(p4_content)
    if review_tax_response:
        try:
            start_tag = "<cluster_table>"
            end_tag = "</cluster_table>"
            start_idx = review_tax_response.find(start_tag) + len(start_tag)
            end_idx = review_tax_response.find(end_tag)
            reviewed_cluster_table_text = review_tax_response[start_idx:end_idx].strip()
            final_taxonomy = reviewed_cluster_table_text
            print("  Taxonomy reviewed and finalized.")
        except Exception as e:
            print(f"  Error parsing reviewed taxonomy: {e}. Using previous version.")
    else:
        print("  Failed to review taxonomy. Using previous version.")

    print("--- Phase 1 Completed ---")
    return final_taxonomy

# --- TnT-LLM Phase 2: LLM-Augmented Text Classification (使用summary生成伪标签) ---
def phase2_classification_sample(summaries_sample, taxonomy):
    """
    执行 TnT-LLM 的第二阶段：使用生成的分类法对样本进行 LLM 标注（生成伪标签）。
    使用summary字段进行分类。
    """
    print("- Starting Phase 2: LLM Annotation for Sample (Pseudo-labeling) -")
    labeled_results = []
    explanation_length = 100
    
    for i, summary in enumerate(summaries_sample):
        # 使用Prompt 5进行分类
        p5_content = prompt5.format(
            cluster_table=taxonomy, 
            input_text=summary, 
            explanation_length=explanation_length
        )
        
        classification_response = call_llm(p5_content)
        if classification_response:
            try:
                # 解析 LLM 的输出
                cat_id_start = "<category-id>"
                cat_id_end = "</category-id>"
                cat_name_start = "<category-name>"
                cat_name_end = "</category-name>"
                exp_start = "<explanation>"
                exp_end = "</explanation>"

                cat_id = classification_response.split(cat_id_start)[1].split(cat_id_end)[0].strip()
                cat_name = classification_response.split(cat_name_start)[1].split(cat_name_end)[0].strip()
                explanation = classification_response.split(exp_start)[1].split(exp_end)[0].strip()

                labeled_results.append({
                    "summary": summary,
                    "category_id": cat_id,
                    "category_name": cat_name,
                    "explanation": explanation
                })
                print(f" LLM Annotated sample {i+1}/{len(summaries_sample)}")
            except Exception as e:
                print(f" Error parsing classification for sample {i+1}: {e}")
                labeled_results.append({
                    "summary": summary,
                    "category_id": "-1",
                    "category_name": "Undefined",
                    "explanation": "Failed to parse LLM response."
                })
        else:
            print(f" Failed to classify sample {i+1}")
            labeled_results.append({
                "summary": summary,
                "category_id": "-1",
                "category_name": "Undefined",
                "explanation": "LLM call failed."
            })

        time.sleep(0.5)
    
    print(f"- Phase 2 (Sample Annotation) Completed - Annotated {len(labeled_results)} samples")
    return labeled_results

# --- 训练轻量级分类器（使用预计算向量） ---
def train_lightweight_classifier_with_precomputed_vectors(labeled_data, precomputed_vectors, model_type='logistic_regression', model_save_path='lightweight_model_with_vectors.pkl'):
    """
    使用预计算的向量训练一个轻量级分类器。
    """
    print(f"- Training Lightweight Classifier with Precomputed Vectors ({model_type}) -")
    
    if len(labeled_data) != len(precomputed_vectors):
        raise ValueError(f"Labeled data length ({len(labeled_data)}) doesn't match vectors length ({len(precomputed_vectors)})")
    
    labels = [item['category_name'] for item in labeled_data]
    
    # 使用预计算的向量作为特征
    X = np.array(precomputed_vectors)
    
    # 确保X是2D数组
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    elif X.ndim == 3:
        X = X.reshape(X.shape[0], -1)
    
    print(f"Training data shape: X={X.shape}, y={len(labels)}")
    print(f"Unique labels: {set(labels)}")
    
    # 统计标签分布
    from collections import Counter
    label_counts = Counter(labels)
    print(f"Label distribution: {dict(label_counts)}")
    
    # 根据指定类型创建和训练模型
    if model_type == 'logistic_regression':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X, labels)
    elif model_type == 'mlp':
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
        model.fit(X, labels)
    elif model_type == 'lightgbm':
        from sklearn.preprocessing import LabelEncoder
        
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(labels)
        
        model = lgb.LGBMClassifier(random_state=42)
        model.fit(X, y_encoded)
        
        label_encoder_save_path = model_save_path.replace('.pkl', '_label_encoder.pkl')
        with open(label_encoder_save_path, 'wb') as f:
            pickle.dump(label_encoder, f)
        print(f"Label encoder saved to {label_encoder_save_path}")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    with open(model_save_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"Model saved to {model_save_path}")
    print("- Lightweight Classifier Training with Precomputed Vectors Completed -")
    return model

# --- 应用轻量级分类器（使用预计算向量） ---
def apply_lightweight_classifier_with_vectors(precomputed_vectors, model, model_type='logistic_regression', label_encoder_path=None):
    """
    使用训练好的轻量级分类器和预计算的向量对数据进行分类。
    """
    print("- Applying Lightweight Classifier with Precomputed Vectors -")
    
    X = np.array(precomputed_vectors)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    elif X.ndim == 3:
        X = X.reshape(X.shape[0], -1)
    
    print(f"Prediction data shape: X={X.shape}")

    predictions = model.predict(X)

    if model_type == 'lightgbm' and label_encoder_path and os.path.exists(label_encoder_path):
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        predictions = label_encoder.inverse_transform(predictions.astype(int))
    elif model_type == 'lightgbm' and not (label_encoder_path and os.path.exists(label_encoder_path)):
        print("Warning: Model type is LightGBM but label encoder path not provided or found. Predictions might be encoded integers.")

    results = []
    for i, pred_label in enumerate(predictions):
        results.append({
            "predicted_category": pred_label,
        })
    
    print(f"- Lightweight Classification with Precomputed Vectors Completed - Classified {len(results)} items")
    return results

# --- 主执行逻辑 ---
def main():
    # 1. 从MongoDB加载数据
    print("步骤1: 加载数据...")
    all_data = load_data_from_mongodb(limit=20000)
    
    if not all_data:
        print("没有加载到数据，程序退出")
        return
    
    # 随机打乱数据
    random.shuffle(all_data)
    
    # 提取数据
    summaries = [item['summary'] for item in all_data]
    vectors = [item['vector'] for item in all_data]
    contents = [item['content'] for item in all_data]
    
    print(f"总数据量: {len(all_data)} 条")
    
    # 2. 采样用于 Phase 1 (Taxonomy Generation) 的数据 (5000条)
    sample_size_phase1 = min(5000, len(summaries))
    summaries_sample_phase1 = summaries[:sample_size_phase1]
    vectors_sample_phase1 = vectors[:sample_size_phase1]
    
    # 3. 采样用于 Phase 2 (Classification Sample for Pseudo-labeling) 的数据 (100条)
    # 从Phase 1之后的数据中取100条
    sample_size_phase2 = min(100, len(summaries) - sample_size_phase1)
    summaries_sample_phase2 = summaries[sample_size_phase1:sample_size_phase1 + sample_size_phase2]
    vectors_sample_phase2 = vectors[sample_size_phase1:sample_size_phase1 + sample_size_phase2]
    
    # 4. 剩余数据用于批量分类
    remaining_summaries = summaries[sample_size_phase1 + sample_size_phase2:]
    remaining_vectors = vectors[sample_size_phase1 + sample_size_phase2:]
    remaining_contents = contents[sample_size_phase1 + sample_size_phase2:]
    
    print(f"Phase 1 数据量: {len(summaries_sample_phase1)} 条")
    print(f"Phase 2 数据量: {len(summaries_sample_phase2)} 条")
    print(f"剩余数据量: {len(remaining_summaries)} 条")
    
    # 5. 执行 Phase 1: 生成分类法
    print("\n" + "="*50)
    print("步骤2: 执行 Phase 1 - 生成暗网分类体系")
    print("="*50)
    taxonomy = phase1_taxonomy_generation(summaries_sample_phase1)
    
    if not taxonomy:
        print("分类体系生成失败，程序退出")
        return
    
    print("\n- 生成的分类体系 -")
    print(taxonomy)
    print("- 分类体系结束 -")
    
    # 保存分类体系
    with open("darkweb_taxonomy.txt", "w", encoding='utf-8') as f:
        f.write(taxonomy)
    print("分类体系已保存到 darkweb_taxonomy.txt")
    
    # 6. 执行 Phase 2: 使用 LLM 对样本进行标注
    print("\n" + "="*50)
    print("步骤3: 执行 Phase 2 - 生成伪标签")
    print("="*50)
    pseudo_labeled_data = phase2_classification_sample(summaries_sample_phase2, taxonomy)
    
    if not pseudo_labeled_data:
        print("伪标签生成失败，程序退出")
        return
    
    # 保存伪标签数据
    with open("pseudo_labels.json", "w", encoding='utf-8') as f:
        json.dump(pseudo_labeled_data, f, indent=4, ensure_ascii=False)
    print(f"伪标签已保存到 pseudo_labels.json，共 {len(pseudo_labeled_data)} 条")
    
    # 7. 训练轻量级分类器
    print("\n" + "="*50)
    print("步骤4: 训练轻量级分类器")
    print("="*50)
    model_type_choice = 'logistic_regression'
    model_save_path = f'lightweight_{model_type_choice}_model_with_vectors.pkl'
    
    lightweight_model = train_lightweight_classifier_with_precomputed_vectors(
        labeled_data=pseudo_labeled_data,
        precomputed_vectors=vectors_sample_phase2,
        model_type=model_type_choice,
        model_save_path=model_save_path
    )
    
    # 8. 应用轻量级分类器到所有数据
    print("\n" + "="*50)
    print("步骤5: 对所有数据应用分类器")
    print("="*50)
    
    # 准备所有需要分类的数据
    all_summaries_for_classification = summaries
    all_vectors_for_classification = vectors
    all_contents_for_classification = contents
    
    print(f"开始对 {len(all_summaries_for_classification)} 条数据进行分类...")
    
    # 分批处理以防止内存问题
    batch_size = 1000
    all_predictions = []
    
    for i in range(0, len(all_summaries_for_classification), batch_size):
        batch_end = min(i + batch_size, len(all_summaries_for_classification))
        batch_vectors = all_vectors_for_classification[i:batch_end]
        
        print(f"分类批次 {i//batch_size + 1}/{(len(all_summaries_for_classification) + batch_size - 1)//batch_size}...")
        
        batch_predictions = apply_lightweight_classifier_with_vectors(
            precomputed_vectors=batch_vectors,
            model=lightweight_model,
            model_type=model_type_choice
        )
        all_predictions.extend(batch_predictions)
    
    print(f"分类完成，共处理 {len(all_predictions)} 条数据")
    
    # 9. 保存完整结果
    print("\n" + "="*50)
    print("步骤6: 保存结果")
    print("="*50)
    
    save_results = []
    for i, (summary, content, prediction) in enumerate(zip(all_summaries_for_classification, all_contents_for_classification, all_predictions)):
        save_result = {
            "id": i + 1,
            "summary": summary,
            "content_preview": content,
            "predicted_category": prediction['predicted_category']
        }
        save_results.append(save_result)
    
    with open("TnT-LLM_darkweb_classification_results.json", "w", encoding='utf-8') as f:
        json.dump(save_results, f, indent=4, ensure_ascii=False)
    print(f"完整分类结果已保存到 TnT-LLM_darkweb_classification_results.json，共 {len(save_results)} 条")
    
    # 10. 统计结果
    print("\n" + "="*50)
    print("步骤7: 结果统计")
    print("="*50)
    
    from collections import Counter
    category_counts = Counter([r['predicted_category'] for r in save_results])
    
    print("分类结果统计:")
    for category, count in category_counts.most_common():
        print(f"  {category}: {count} 条 ({count/len(save_results)*100:.1f}%)")
    
    print(f"未分类数量: {category_counts.get('Undefined', 0)} 条")
    print(f"分类失败数量: {category_counts.get('-1', 0)} 条")
    
    print("\n✅ 所有步骤完成！")

if __name__ == "__main__":
    main()