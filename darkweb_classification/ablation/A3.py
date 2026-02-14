import os
import pymongo
import re
import time
import json
import numpy as np
from bs4 import BeautifulSoup
from openai import OpenAI
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any, Tuple
from scipy.spatial.distance import cosine
from bson import ObjectId

# 导入机器学习相关库
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


class TaxonomyBuilder:
    
    def __init__(self):
        # 初始化配置
        self.config = {
            "mongo_host": "mongodb://192.168.31.9:27017/",
            "db_name": "3our_spider_db",
            "collection_name": "content20000",
            "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
            "openai_base_url": "https://api.chatanywhere.tech/v1    ",
            "max_summary_length": 100,
            "max_text_length": 8000,
            "batch_size": 40,
            "request_delay": 1,
            "taxonomy_batch_size": 500,
            "max_taxonomy_updates": 10,
            "distance_threshold": 0.8,
            "min_samples_for_centroid": 3,
        }
        
        # 初始化客户端
        self.client = None
        self.db = None
        self.collection = None
        self.openai_client = None
        
        # 主题管理
        self.all_summaries = []
        self.final_taxonomy = []  # List[{"category": "..."}]
        
        # 向量管理
        self.category_centroids = {}
        self.category_vectors = defaultdict(list)
        self.document_vectors = {}
        
        # 机器学习模型
        self.ml_model = None
        self.label_encoder = None
        
        self._init_clients()
    
    def _init_clients(self):
        """初始化数据库和OpenAI客户端"""
        try:
            self.client = pymongo.MongoClient(self.config["mongo_host"])
            self.db = self.client[self.config["db_name"]]
            self.collection = self.db[self.config["collection_name"]]
            print("MongoDB连接成功")
        except Exception as e:
            print(f"MongoDB连接失败: {e}")
            raise
        
        try:
            self.openai_client = OpenAI(
                api_key=self.config["openai_api_key"],
                base_url=self.config["openai_base_url"].strip()
            )
            print("OpenAI客户端初始化成功")
        except Exception as e:
            print(f"OpenAI客户端初始化失败: {e}")
            raise

    def _migrate_category_vectors(self, merged_from: Dict[str, List[str]]):
        """
        根据 merged_from 映射，将旧类别的向量迁移到新类别，并清理旧键。
        """
        new_vectors = defaultdict(list)
        
        # 初始化新类别容器（保留可能已存在的向量）
        for new_cat in merged_from.keys():
            if new_cat in self.category_vectors:
                new_vectors[new_cat] = self.category_vectors[new_cat].copy()
            else:
                new_vectors[new_cat] = []
        
        # 迁移旧向量
        for new_cat, old_list in merged_from.items():
            for old_cat in old_list:
                if old_cat in self.category_vectors:
                    new_vectors[new_cat].extend(self.category_vectors[old_cat])
        
        # 替换为新结构
        self.category_vectors = {cat: vecs for cat, vecs in new_vectors.items() if vecs}

    def stage1_summarization(self, sample_size: int = None) -> List[Dict]:
        """
        第一阶段：**直接读取已有的 summary 和 vector**（跳过摘要生成）
        """
        try:
            if sample_size is None:
                cursor = self.collection.find(
                    {"summary": {"$exists": True, "$ne": "", "$ne": "内容过短，无法生成摘要"}},
                    no_cursor_timeout=True
                )
                total_count = self.collection.count_documents({
                    "summary": {"$exists": True, "$ne": "", "$ne": "内容过短，无法生成摘要"}
                })
                print(f"准备处理全部 {total_count} 个已有摘要的文档...")
            else:
                cursor = self.collection.find(
                    {"summary": {"$exists": True, "$ne": "", "$ne": "内容过短，无法生成摘要"}},
                    no_cursor_timeout=True
                ).limit(sample_size)
                total_count = min(sample_size, self.collection.count_documents({
                    "summary": {"$exists": True, "$ne": "", "$ne": "内容过短，无法生成摘要"}
                }))
                print(f"准备处理最多 {total_count} 个已有摘要的文档...")
        except Exception as e:
            print(f"获取文档失败: {e}")
            return []

        results = []
        processed = 0

        try:
            for doc in cursor:
                try:
                    summary = doc.get("summary", "").strip()
                    vector = doc.get('vector')
                    
                    if not summary or summary in ["内容过短，无法生成摘要"]:
                        print(f"文档 {doc.get('_id', '未知')} 摘要无效，跳过")
                        continue

                    if vector is None:
                        print(f"文档 {doc.get('_id', '未知')} 缺少 'vector'，跳过")
                        continue
                    
                    if isinstance(vector, list):
                        vector = np.array(vector).flatten()
                    else:
                        print(f"文档 {doc.get('_id', '未知')} 向量格式异常，跳过")
                        continue

                    result = {
                        "doc_id": str(doc['_id']),
                        "summary": summary,
                        "vector": vector,
                        "processed_at": datetime.now()
                    }
                    
                    results.append(result)
                    self.all_summaries.append(summary)
                    self.document_vectors[str(doc['_id'])] = vector
                    
                    processed += 1
                    if processed % 10 == 0:
                        print(f"已加载 {processed}/{total_count} 个文档")

                    if (processed) % self.config["batch_size"] == 0:
                        time.sleep(self.config["request_delay"])

                except Exception as e:
                    print(f"处理文档 {doc.get('_id', '未知')} 时出错: {e}")
                    continue

        finally:
            cursor.close()
        
        print(f"第一阶段完成，共成功加载 {len(results)} 个文档的摘要与向量")
        return results

    def calculate_category_centroids(self):
        print("开始计算类别质心...")
        self.category_centroids.clear()
        for category, vectors in self.category_vectors.items():
            if len(vectors) >= self.config["min_samples_for_centroid"]:
                centroid = np.mean(np.array(vectors), axis=0)
                self.category_centroids[category] = centroid
                print(f"   '{category}' → {len(vectors)} 样本 → 质心维度 {len(centroid)}")
            else:
                print(f"   '{category}' 样本不足 ({len(vectors)} < {self.config['min_samples_for_centroid']})，跳过质心计算")

    def find_best_category_by_distance(self, vector: np.ndarray) -> Tuple[str, float]:
        if not self.category_centroids:
            return "其他", float('inf')
        
        min_distance = float('inf')
        best_category = "其他"
        
        for category, centroid in self.category_centroids.items():
            try:
                dist = cosine(vector, centroid)
                if dist < min_distance:
                    min_distance = dist
                    best_category = category
            except Exception as e:
                print(f"计算与 '{category}' 距离出错: {e}")
        
        return best_category, min_distance

    def assess_potential_new_category(self, text: str, vector: np.ndarray) -> Dict[str, Any]:
        try:
            existing_cats = list(self.category_centroids.keys())
            prompt = f"""
                You are a senior dark web threat intelligence analyst.

                # Background
                - You are reviewing a new dark web text snippet.
                - You have a predefined set of dark web categories (representing distinct domains of dark web content).
                - Your task is to determine whether this text belongs to an existing category or represents a genuinely new category.

                # Rules
                - **Do not alter existing categories**: Only mark as a match if the text **clearly belongs** to one of the existing dark web categories.
                - **Do not force a classification**: If the text describes a distinct threat domain not covered by current categories, propose a new one.
                - **Any new category must**:
                  - Be a **2–4 word name phrase**
                  - Represent a **broad but concrete dark web domain** (e.g., "Illicit Firearms Trade")
                  - Use **professional, legally compliant terminology** (e.g., "Child Sexual Abuse Material", not "Child Porn")
                  - Avoid **generic terms**: "Services", "Content", "Platform", "Activities", "Other", "Miscellaneous"
                  - Be **specific enough to guide classification**, yet **not overly narrow** (e.g., use "Payment Card Fraud" instead of "Stolen Visa Cards")

                # Existing Top-Level Categories
                {json.dumps(existing_cats, ensure_ascii=False)}

                # New Text
                {text}

                # Output
                Output ONLY a valid JSON object containing:
                - "fits_existing": true if the text clearly belongs to an existing category, false otherwise.
                - "suggested_category": if "fits_existing" is true, provide the **exact name** of the matching existing category; if false, provide a new top-level category name (2–4 words).
                - "reasoning": a brief one-sentence justification.

                Do not include any other text, markdown, or formatting.
                """
            time.sleep(self.config["request_delay"])
            response = self.openai_client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": "You are a topic classification expert."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"新类别评估失败: {e}")
            return {"fits_existing": True, "suggested_category": "", "reasoning": "评估出错"}

    def _handle_new_category_addition(self, new_category: str, representative_vector: np.ndarray, taxonomy: List[Dict]):
        new_category = new_category.strip()
        existing_cats = {item["category"] for item in taxonomy}
        if not new_category or new_category in existing_cats:
            return
        new_item = {"category": new_category}
        taxonomy.append(new_item)
        self.category_vectors[new_category] = [representative_vector]
        print(f"成功添加新类别: '{new_category}'")

    def generate_initial_taxonomy(self, batch_summaries: List[str]) -> List[Dict]:
        try:
            prompt = f"""
            # Instruction
            Generate a taxonomy from dark web content summaries for classifying dark web texts. The taxonomy must be accurate, mutually exclusive, and comprehensive.

            # Context
            You are a senior dark web intelligence analyst. Your task is to derive a concise set of dark web content categories from a batch of textual summaries. These summaries describe the content found on dark web pages.

            # Requirements
            - Each category must be a **2–4 word name phrase** representing a **broad but concrete** thematic domain.
            - Categories must be **mutually exclusive**—no overlap or contradiction.
            - Output must be in **English only**.
            - Strike the right granularity:
              • GOOD examples: "Payment Card Fraud", "Illicit Drug Trade", "Malware Distribution"
              • TOO BROAD: "Illegal Activity", "Cybercrime", "Dark Web"
              • TOO NARROW: "Stolen Visa Cards", "Forum User Guide", "Marketplace Reviews", "Error 404 Page"
            - STRICTLY AVOID:
              - Generic filler terms: "Services", "Content", "Information", "Platform", "Activities", "Other", "Miscellaneous"
              - Specific items (e.g., "Passports", "Cocaine", "Ransomware") — instead, abstract to their **domain**: "Identity Fraud", "Illicit Drug Trade", "Malware & Exploits"
            - The resulting taxonomy should **closely reflect the input data**: do not omit important categories or introduce irrelevant ones.
            - Categories must be **specific, meaningful**, and grounded in the data—do not invent categories not supported by the summaries.
            - The taxonomy should effectively serve the purpose of dark web content classification.

            # Data
            {json.dumps(batch_summaries, indent=2)}

            # Output
            Output ONLY valid JSON with a top-level key "taxonomy" containing a list of objects in the form {{"category": "..."}}.  
            Do not include any other text, explanations, or formatting.
            """
            time.sleep(self.config["request_delay"])
            response = self.openai_client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": "You are a precision-focused dark web taxonomy expert."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            result = json.loads(response.choices[0].message.content)
            taxonomy = result.get("taxonomy", [])
            clean_taxonomy = [{"category": item["category"]} for item in taxonomy if "category" in item]
            print(f"Initial taxonomy generated: {len(clean_taxonomy)} top-level categories")
            return clean_taxonomy
        except Exception as e:
            print(f"Failed to generate initial taxonomy: {e}")
            return []

    def update_taxonomy(self, current_taxonomy: List[Dict], batch_summaries: List[str]) -> Tuple[List[Dict], Dict[str, List[str]]]:
        try:
            current_cats = [item["category"] for item in current_taxonomy]
            prompt = f"""
            You are a senior dark web intelligence analyst updating a flat threat taxonomy (top-level categories only).
            
            # Background
            - You are given a current taxonomy and a batch of new dark web content summaries.
            - Your task is to:  
              (1) evaluate the quality of the current taxonomy against the new data. 
              (2) rate it on a scale from 0 to 100,  
              (3) propose modifications if necessary, and  
              (4) output an improved flat taxonomy.
            
            # Rules
            - **Granularity**: Each category must be a 2–4 word phrase representing a broad but concrete criminal ecosystem.  
              • GOOD examples: "Payment Card Fraud", "Illicit Drug Trade"  
              • TOO BROAD: "Cybercrime", "Illegal Activity"  
              • TOO NARROW: "Stolen Visa Cards", "Forum Rules"
            - **Merge similar categories** (e.g., "Fake IDs" + "Stolen SSNs" → "Identity Fraud").
            - **Add new categories only if** the new summaries reveal a clear, recurring threat domain not covered by the current taxonomy.
            - **Remove** categories that are vague (e.g., "Other", "Services"), redundant, or unsupported by the data.
            - **Never use**: "Other", "Miscellaneous", "Services", "Content", "Platform", "Dark Web", "General", "Undefined".
            - **Output must be a flat list** — no hierarchy, no subcategories.
            
            # CRITICAL: Output Mapping
            For every category in your updated taxonomy, explicitly list **all categories from the CURRENT taxonomy that were merged into it**.
            - If a category is unchanged, map it to itself.
            - This enables correct migration of historical document vectors.

            # Evaluation Criteria
            ## Intrinsic Quality
            - Are category names clear, consistent, and mutually exclusive?
            - Are they relevant to cybersecurity threat intelligence?
            - Do they contain vague or prohibited terms?
            
            ## Extrinsic Quality
            - Can the taxonomy accurately and unambiguously classify the new summaries?
            - Are there missing threat domains in the current taxonomy?
            - Are there redundant or data-unsupported categories?
            
            # Current Taxonomy
            {json.dumps(current_cats, indent=2, ensure_ascii=False)}
            
            # New Summaries
            {json.dumps(batch_summaries, indent=2, ensure_ascii=False)}
            
            # Output Instructions
            Output ONLY a valid JSON object with the following keys:
            - "rating": an integer from 0 to 100 (higher = better quality)
            - "explanation": a string (≤ 50 words) explaining the rating
            - "suggestion": a string (≤ 30 words) describing necessary edits
            - "updated_taxonomy": a list of objects in the form {{"category": "Consolidated Category Name"}}
            - "merged_from": {{
                  "New Cat A": ["Old Cat 1", "Old Cat 2"],
                  "Unchanged Cat": ["Unchanged Cat"],
                  ...
              }}
            - Keys in "merged_from" must be from "updated_taxonomy".
            - Values must be subsets of the input "Current Taxonomy".
            - Do not include any other text, markdown, or formatting.
            """
            time.sleep(self.config["request_delay"])
            response = self.openai_client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": "You are a cyber threat intelligence taxonomy optimizer."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            result = json.loads(response.choices[0].message.content)
            updated = result.get("updated_taxonomy", current_taxonomy)
            merged_from = result.get("merged_from", {})
            
            clean_updated = [{"category": item["category"]} for item in updated if "category" in item]
            clean_merged = {}
            for new_cat, old_list in merged_from.items():
                if isinstance(old_list, list):
                    clean_merged[new_cat] = [str(x).strip() for x in old_list if x and isinstance(x, str)]
            
            print(f"Taxonomy updated: {len(clean_updated)} top-level categories")
            return clean_updated, clean_merged
        except Exception as e:
            print(f"Taxonomy update failed: {e}")
            # 回退：无变更，自映射
            fallback_merged = {item["category"]: [item["category"]] for item in current_taxonomy}
            return current_taxonomy, fallback_merged

    def final_review_taxonomy(self, taxonomy: List[Dict]) -> Tuple[List[Dict], Dict[str, List[str]]]:
        try:
            cat_names = [item["category"] for item in taxonomy]
            prompt = f"""
            # Task Instruction
            You are a senior dark web intelligence analyst responsible for performing the final review of a **dark web taxonomy**.

            # Background Requirements
            - The output will be used in an **automated dark web classification system**.
            - The taxonomy must adhere to **industry-standard terminology**, such as: "Pornographic Content", "Illegal Marketplaces", "Hacking Services", etc.

            # Rules
            - Each category must be a **2–4 word name phrase** representing a **broad but concrete** thematic domain.
            - Categories must be **mutually exclusive**—no overlap or contradiction.
            - Output must be in **English only**.
            - Strike the right granularity:
              • GOOD examples: "Payment Card Fraud", "Illicit Drug Trade", "Malware Distribution"
              • TOO BROAD: "Illegal Activity", "Cybercrime", "Dark Web"
              • TOO NARROW: "Stolen Visa Cards", "Forum User Guide", "Marketplace Reviews", "Error 404 Page"
            - STRICTLY AVOID:
              - Generic filler terms: "Services", "Content", "Information", "Platform", "Activities", "Other", "Miscellaneous"
              - Specific items (e.g., "Passports", "Cocaine", "Ransomware") — instead, abstract to their **domain**: "Identity Fraud", "Illicit Drug Trade", "Malware & Exploits"
            - The resulting taxonomy should **effectively serve dark web content classification**.
            - Categories must be **specific, meaningful**, and grounded in real-world threat intelligence.

            # Current Categories
            {json.dumps(cat_names, indent=2, ensure_ascii=False)}

            # Output
            Output ONLY a valid JSON object with the following structure:
            {{
                "final_taxonomy": [
                    {{"category": "Consolidated Category Name"}},
                    ...
                ],
                "merged_from": {{
                    "Consolidated Category Name": ["Original Category A", "Original Category B", ...],
                    ...
                }}
            }}
            - Every category in "final_taxonomy" must appear as a key in "merged_from".
            - All values in "merged_from" must be subsets of the input categories.
            - Do not include any other text, markdown, or formatting.
            """
            time.sleep(self.config["request_delay"])
            response = self.openai_client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": "You are a precision-focused cyber intelligence taxonomy auditor."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            result = json.loads(response.choices[0].message.content)
            final = result.get("final_taxonomy", taxonomy)
            merged_from = result.get("merged_from", {})
            
            clean_final = [{"category": item["category"]} for item in final if "category" in item]
            clean_merged = {}
            for new_cat, old_list in merged_from.items():
                if isinstance(old_list, list):
                    clean_merged[new_cat] = [str(x).strip() for x in old_list if x and isinstance(x, str)]
            
            print(f"Final taxonomy: {len(clean_final)} top-level categories")
            return clean_final, clean_merged
        except Exception as e:
            print(f"Final review failed: {e}")
            # 回退：自映射
            fallback_merged = {item["category"]: [item["category"]] for item in taxonomy}
            return taxonomy, fallback_merged

    def find_best_topic_match(self, text: str, topics: List[str]) -> str:
        if not topics:
            return "其他"
        try:
            prompt = f"""
                You are a senior dark web threat intelligence analyst.

                Task: Assign the input text to the SINGLE most appropriate category from the provided list.  
                - ONLY use an **exact name** from the "Categories" list below.  
                - Return **"Other"** ONLY if the text clearly does NOT belong to ANY of the listed categories (e.g., neutral content like "Bible Study", "Error 404", "Privacy Guide").  
                - NEVER paraphrase, merge, or invent categories.  
                - If the text relates to illegal or illicit activity, it almost certainly fits one of the threat categories — do NOT default to "Other".

                Text: {text}

                Categories: {', '.join(topics)}

                Output ONLY the category name or "Other". No explanation, no punctuation, no extra text.
                """
            time.sleep(self.config["request_delay"])
            response = self.openai_client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": "You are a precise classifier. Output ONLY the exact category name or 'Others'."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            result = response.choices[0].message.content.strip().strip('"').strip('"')
            return result if result in topics else "其他"
        except Exception as e:
            print(f"主题匹配失败，回退到'其他': {e}")
            return "其他"

    def assign_categories_to_documents(self, documents: List[Dict], taxonomy: List[Dict]) -> List[Dict]:
        level1_cats = [item["category"] for item in taxonomy]
        for doc in documents:
            summary = doc["summary"]
            cat = self.find_best_topic_match(summary, level1_cats)
            doc["category"] = cat
        return documents

    def stage2_taxonomy_building(self, documents: List[Dict]) -> List[Dict]:
        if not documents:
            return []

        batch_size = self.config["taxonomy_batch_size"]
        batches = [documents[i:i+batch_size] for i in range(0, len(documents), batch_size)]
        print(f"开始构建一级分类法（共 {len(batches)} 批）...")

        taxonomy = []

        for idx, batch in enumerate(batches):
            summaries = [d["summary"] for d in batch]
            if idx == 0:
                taxonomy = self.generate_initial_taxonomy(summaries)
            else:
                taxonomy, merged_from = self.update_taxonomy(taxonomy, summaries)
                self._migrate_category_vectors(merged_from)

            # 👇 统一初始化：确保 taxonomy 中所有类别都在 category_vectors 中
            for item in taxonomy:
                cat = item["category"]
                if cat not in self.category_vectors:
                    self.category_vectors[cat] = []
            
            self.assign_categories_to_documents(batch, taxonomy)

            for doc in batch:
                cat = doc.get("category")
                vec = doc.get("vector")
                if cat and cat != "其他" and vec is not None:
                    self.category_vectors[cat].append(vec)

            self.calculate_category_centroids()

            for doc in batch:
                cat = doc.get("category")
                vec = doc.get("vector")
                if cat == "其他" or vec is None:
                    continue
                centroid = self.category_centroids.get(cat)
                if centroid is None:
                    continue
                dist = cosine(vec, centroid)
                if dist > self.config["distance_threshold"]:
                    print(f"文档 {doc['doc_id']} 距离质心过大 ({dist:.3f})，评估是否需新类别...")
                    llm_res = self.assess_potential_new_category(doc["summary"], vec)
                    if not llm_res.get("fits_existing", True):
                        new_cat = llm_res.get("suggested_category", "").strip()
                        if new_cat:
                            self._handle_new_category_addition(new_cat, vec, taxonomy)
                            self.calculate_category_centroids()

            print(f"批次 {idx+1}/{len(batches)} 处理完成")

        final_taxonomy, merged_from = self.final_review_taxonomy(taxonomy)
        self._migrate_category_vectors(merged_from)
        self.final_taxonomy = [{"category": item["category"]} for item in final_taxonomy if "category" in item]
        return self.final_taxonomy

    def export_to_json(self, documents: List[Dict], output_path: str = "output/full_taxonomy_results.json"):
        try:
            output_data = []
            for doc in documents:
                summary = doc.get("summary", "").strip()
                if not summary or summary == "内容过短，无法生成摘要":
                    continue
                category = doc.get("category", "未分类").strip()
                output_data.append({
                    "alert": summary,
                    "category": category
                })
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            print(f"分类结果已导出为 JSON 文件: {output_path}")
        except Exception as e:
            print(f"JSON 导出失败: {e}")

    def taxonomy_to_cluster_table(self, taxonomy, max_num_clusters=None):
        # Extract category names
        categories = [item["category"] for item in taxonomy if "category" in item]
        
        # Apply max_num_clusters if specified
        if max_num_clusters is not None:
            categories = categories[:max_num_clusters]
        
        # Build markdown table lines
        lines = [
            "|id|name|description|",
            "|-|-|-|"
        ]
        
        for idx, name in enumerate(categories, start=1):
            # Escape any pipe characters in name to avoid breaking markdown table
            safe_name = str(name).replace("|", "\\|")
            lines.append(f"|{idx}|{safe_name}| |")
        
        return "\n".join(lines)
    
    def phase2_classification_sample(self, alerts_to_classify_sample, cluster_table):
        """
        执行 TnT-LLM 的第二阶段：使用生成的分类法对一个样本进行 LLM 标注（生成伪标签）。
        """
        print("- Starting Phase 2: LLM Annotation for Sample (Pseudo-labeling) -")
        labeled_results = []
        explanation_length = 100
        
        for i, alert in enumerate(alerts_to_classify_sample):
            # 使用 Prompt 进行分类
            prompt = """
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
            prompt = prompt.format(cluster_table=cluster_table, input_text=alert, explanation_length=explanation_length)
            
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-5-mini",
                    messages=[
                        {"role": "system", "content": "You are a precise classifier."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.6
                )
                classification_response = response.choices[0].message.content.strip().strip('"').strip('"')
            except Exception as e:
                print(f"LLM调用失败: {e}")
                classification_response = None

            if classification_response:
                try:
                    # 解析 LLM 的输出，提取 <category-id>, <category-name>, <explanation>
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
                        "original_alert": alert,
                        "category_id": cat_id,
                        "category_name": cat_name,
                        "explanation": explanation
                    })
                    print(f" LLM Annotated sample {i+1}/{len(alerts_to_classify_sample)}")
                except Exception as e:
                    print(f" Error parsing classification for sample {i+1}: {e}")
                    labeled_results.append({
                        "original_alert": alert,
                        "category_id": "-1",
                        "category_name": "Undefined",
                        "explanation": "Failed to parse LLM response."
                    })
            else:
                print(f" Failed to classify sample {i+1}")
                labeled_results.append({
                    "original_alert": alert,
                    "category_id": "-1",
                    "category_name": "Undefined",
                    "explanation": "LLM call failed."
                })

        print("- Phase 2 (Sample Annotation) Completed -")
        return labeled_results
    
    def train_lightweight_classifier_with_precomputed_vectors(self, labeled_data, precomputed_vectors, model_type='logistic_regression', model_save_path='lightweight_model_with_vectors.pkl'):
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

        # 根据指定类型创建和训练模型
        if model_type == 'logistic_regression':
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X, labels)
        elif model_type == 'mlp':
            from sklearn.neural_network import MLPClassifier
            model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
            model.fit(X, labels)
        elif model_type == 'lightgbm':
            import lightgbm as lgb
            from sklearn.preprocessing import LabelEncoder
            
            # LightGBM 需要将标签转换为数值
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(labels)
            
            # 使用LightGBM sklearn API
            model = lgb.LGBMClassifier(random_state=42)
            model.fit(X, y_encoded)
            
            # 保存标签编码器
            label_encoder_save_path = model_save_path.replace('.pkl', '_label_encoder.pkl')
            with open(label_encoder_save_path, 'wb') as f:
                pickle.dump(self.label_encoder, f)
            print(f"Label encoder saved to {label_encoder_save_path}")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # 保存模型
        with open(model_save_path, 'wb') as f:
            pickle.dump(model, f)

        print(f"Model saved to {model_save_path}")
        print("- Lightweight Classifier Training with Precomputed Vectors Completed -")
        return model
    
    def apply_lightweight_classifier_with_vectors(self, precomputed_vectors, model, model_type='logistic_regression', label_encoder_path=None):
        """
        使用训练好的轻量级分类器和预计算的向量对数据进行分类。
        """
        print("- Applying Lightweight Classifier with Precomputed Vectors -")
        
        # 使用预计算的向量作为特征
        X = np.array(precomputed_vectors)
        
        # 确保X是2D数组
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        
        print(f"Prediction data shape: X={X.shape}")

        predictions = model.predict(X)

        # 如果是 LightGBM，需要将预测的编码转回原始标签名
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
                "explanation": "Generated by lightweight classifier with precomputed vectors"
            })
        print("- Lightweight Classification with Precomputed Vectors Completed -")
        return results

    def classify_all_documents_ml(self, taxonomy: List[Dict]):
        """
        使用机器学习模型（逻辑回归）对全量文档进行分类
        """
        print("=== 开始使用机器学习模型对全量文档进行分类 ===")
        
        # 1. 获取所有含有效summary的文档
        try:
            cursor = self.collection.find(
                {"summary": {"$exists": True, "$ne": "", "$ne": "内容过短，无法生成摘要"}},
                {"_id": 1, "summary": 1, "vector": 1}
            )
            docs_list = list(cursor)
            total = len(docs_list)
            print(f"共找到 {total} 个含有效 summary 的文档")
        except Exception as e:
            print(f"查询全量文档失败: {e}")
            return

        if not docs_list:
            print("没有找到有效的文档进行分类")
            return

        # 提取文档内容、向量和ID
        summaries = [doc.get("summary", "") for doc in docs_list]
        vectors = []
        doc_ids = []
        
        for doc in docs_list:
            vector = doc.get("vector")
            if vector is not None:
                if isinstance(vector, list):
                    vector = np.array(vector).flatten()
                vectors.append(vector)
                doc_ids.append(doc["_id"])
            else:
                # 如果文档没有向量，暂时跳过
                vectors.append(None)
                doc_ids.append(doc["_id"])

        # 过滤掉没有向量的文档
        valid_indices = [i for i, v in enumerate(vectors) if v is not None]
        valid_summaries = [summaries[i] for i in valid_indices]
        valid_vectors = [vectors[i] for i in valid_indices]
        valid_doc_ids = [doc_ids[i] for i in valid_indices]

        if not valid_vectors:
            print("没有找到有效的向量进行分类")
            return

        print(f"共有 {len(valid_vectors)} 个文档具有有效向量，准备使用LLM生成小数据集并训练分类器...")

        # 2. 从有效文档中采样一部分用于LLM标注
        sample_size = min(100, len(valid_summaries))  # 根据实际情况调整样本大小
        sampled_summaries = valid_summaries[:sample_size]
        sampled_vectors = valid_vectors[:sample_size]
        sampled_doc_ids = valid_doc_ids[:sample_size]

        print(f"使用 {sample_size} 个文档样本请求LLM生成标注数据...")

        # 3. 使用 LLM 对样本进行标注
        cluster_table = self.taxonomy_to_cluster_table(taxonomy)
        pseudo_labeled_data = self.phase2_classification_sample(sampled_summaries, cluster_table)

        # 4. 训练逻辑回归分类器
        model_type_choice = 'logistic_regression'
        model_save_path = f'lightML_{model_type_choice}_model_with_vectors.pkl'
        
        self.ml_model = self.train_lightweight_classifier_with_precomputed_vectors(
            labeled_data=pseudo_labeled_data,
            precomputed_vectors=sampled_vectors[:len(pseudo_labeled_data)],  # 只使用有标签的数据对应的向量
            model_type=model_type_choice,
            model_save_path=model_save_path
        )

        # 5. 应用逻辑回归分类器对所有文档进行分类
        print(f"应用逻辑回归分类器对 {len(valid_summaries)} 个文档进行分类...")
        final_results = self.apply_lightweight_classifier_with_vectors(
            precomputed_vectors=valid_vectors,
            model=self.ml_model,
            model_type=model_type_choice
        )
        print(f"分类完成，共处理 {len(final_results)} 个文档")

        # 6. 更新数据库中的分类结果
        batch_for_update = []
        for i, result in enumerate(final_results):
            if i < len(valid_doc_ids):
                batch_for_update.append(
                    pymongo.UpdateOne(
                        {"_id": valid_doc_ids[i]},
                        {
                            "$set": {
                                "category": result['predicted_category'],
                                "classification_updated_at": datetime.now()
                            }
                        }
                    )
                )
            
            # 批量更新数据库
            if len(batch_for_update) >= 100:
                try:
                    self.collection.bulk_write(batch_for_update, ordered=False)
                    batch_for_update = []
                except Exception as e:
                    print(f"批量更新失败: {e}")

        # 处理剩余更新
        if batch_for_update:
            try:
                self.collection.bulk_write(batch_for_update, ordered=False)
            except Exception as e:
                print(f"最后一批更新失败: {e}")

        # 7. 导出结果到JSON文件
        save_results = []
        for i, result in enumerate(final_results):
            if i < len(valid_summaries):
                save_result = {
                    "alert": valid_summaries[i],
                    "category": result['predicted_category']
                }
                save_results.append(save_result)

        # 保存完整结果到文件
        os.makedirs("output", exist_ok=True)
        output_path = "output/full_taxonomy_results_LR.json"
        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(save_results, f, indent=4, ensure_ascii=False)
        print(f"完整分类结果已保存到 {output_path}")
        
        return save_results

    def classify_all_documents(self, taxonomy: List[Dict]):
        """
        包装函数，可以选择使用哪种分类方法
        """
        # 使用机器学习方法进行分类
        return self.classify_all_documents_ml(taxonomy)

    def run_analysis(self, sample_size_for_taxonomy: int = 500):
        print("=== 第一阶段：基于抽样构建一级分类体系（使用已有 summary）===")
        docs_sample = self.stage1_summarization(sample_size=sample_size_for_taxonomy)

        if not docs_sample:
            print("无有效样本，无法构建 taxonomy")
            return

        print("\n=== 第二阶段：构建一级 taxonomy ===")
        taxonomy = self.stage2_taxonomy_building(docs_sample)
        self.final_taxonomy = taxonomy

        print("\n=== 第三阶段：使用机器学习模型对全量文档进行一级分类 ===")
        classified_docs = self.classify_all_documents(taxonomy)

        print("\n全流程完成！")

        # 打印统计信息
        self._print_results(classified_docs)

    def _print_results(self, documents: List[Dict]):
        print("\n" + "="*60)
        print("最终分析报告")
        print("="*60)
        print(f"总处理文档数: {len(documents)}")
        print(f"一级主题数: {len(self.final_taxonomy)}")

        dist = defaultdict(int)
        for d in documents:
            dist[d.get('category', '未分类')] += 1
        for topic, cnt in sorted(dist.items(), key=lambda x: -x[1]):
            print(f"  • {topic}: {cnt} 篇")


def main():
    analyzer = TaxonomyBuilder()
    try:
        analyzer.run_analysis(sample_size_for_taxonomy=5000)
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n程序异常终止: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()