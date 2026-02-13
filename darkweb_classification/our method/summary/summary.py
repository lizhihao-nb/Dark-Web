import os
import pymongo
import re
import time
from bs4 import BeautifulSoup
from openai import OpenAI


def extract_clean_text(html_content: str) -> str:
    try:
        soup = BeautifulSoup(html_content, 'lxml')
        for tag in soup(['script', 'style', 'meta', 'link', 'nav', 'footer', 'header']):
            tag.decompose()
        text = soup.get_text(separator=' ', strip=True)
        text = re.sub(r'\s+', ' ', text).strip()
        return text if text else ""
    except Exception:
        return re.sub(r'<[^>]+>', '', html_content)


def main():
    # 配置
    mongo_host = "mongodb://192.168.31.9:27017/"
    db_name = "3our_spider_db"
    collection_name = "content20000"
    openai_api_key = os.getenv("OPENAI_API_KEY", "sk-cmz5LsPuRvfGFw9jhMa5Q89hoDVUoQYNaugjbX3zDIRDtIn6")
    openai_base_url = "https://api.chatanywhere.tech/v1"
    max_text_length = 8000
    request_delay = 1

    # 初始化客户端
    client = pymongo.MongoClient(mongo_host)
    db = client[db_name]
    collection = db[collection_name]
    openai_client = OpenAI(api_key=openai_api_key, base_url=openai_base_url.strip())
    
    print("🔍 开始生成摘要...")

    # 只处理尚未有 summary 的文档
    query = {"summary": {"$exists": False}}
    total = collection.count_documents(query)
    print(f"📌 共有 {total} 条文档需要生成摘要")

    if total == 0:
        print("✅ 无需处理")
        return

    cursor = collection.find(query, no_cursor_timeout=True)
    batch_updates = []
    success = 0
    failed = 0
    batch_size = 100

    try:
        for doc in cursor:
            try:
                content = doc.get('content')
                if not content:
                    failed += 1
                    continue

                plain_text = extract_clean_text(content)
                if not plain_text or len(plain_text.strip()) < 50:
                    summary = "内容过短，无法生成摘要"
                else:
                    if len(plain_text) > max_text_length:
                        plain_text = plain_text[:max_text_length] + "...[文本已截断]"

                    time.sleep(request_delay)
                    response = openai_client.chat.completions.create(
                        model="deepseek-v3",
                        messages=[
                            {
                               f"""
                               # Instruction
                                Generate a precise summary (within 100 words) for the following dark web text.
                                
                                # Context
                                You are a professional dark web content analyst skilled in generating concise and accurate text summaries.
                                
                                # Requirements
                                Adhere to the following principles:
                                1. Extract the core content of the text while ignoring irrelevant details
                                2. Maintain objectivity and neutrality without adding subjective evaluations
                                3. Highlight key information and central themes
                                4. Ensure the summary length does not exceed 100 words
                                5. For sensitive or illegal content, provide factual descriptions without moral judgment

                        Text content:
                        {plain_text}

                        Generate a concise and accurate summary:"""
                            }
                        ],
                        temperature=0.3
                    )
                    summary = response.choices[0].message.content.strip()
                    if len(summary.split()) > 100:
                        summary = ' '.join(summary.split()[:100]) + "..."

                batch_updates.append(
                    pymongo.UpdateOne(
                        {"_id": doc["_id"]},
                        {"$set": {"summary": summary}}
                    )
                )
                success += 1

                # 每 10 条打印一次进度
                if (success + failed) % 10 == 0:
                    print(f"🔄 进度: {success + failed} / {total} (成功: {success}, 失败: {failed})")

                # 批量提交
                if len(batch_updates) >= batch_size:
                    collection.bulk_write(batch_updates, ordered=False)
                    batch_updates = []

            except Exception as e:
                failed += 1
                print(f"⚠️  文档 {doc.get('_id')} 处理失败: {e}")

        # 提交剩余
        if batch_updates:
            collection.bulk_write(batch_updates, ordered=False)

        print("\n✅ 摘要生成完成！")
        print(f"📊 成功: {success} | 失败/跳过: {failed} | 总计: {success + failed}")

    finally:
        cursor.close()


if __name__ == "__main__":
    main()