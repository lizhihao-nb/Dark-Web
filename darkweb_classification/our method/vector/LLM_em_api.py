import os
import pymongo
from bs4 import BeautifulSoup
from openai import OpenAI, RateLimitError, APIConnectionError
import time
from datetime import datetime
from bson import ObjectId
import logging  # 新增日志记录模块

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 1. 连接MongoDB
client = pymongo.MongoClient("mongodb://192.168.31.9:27017/")
db = client["3our_spider_db"]
collection = db["content"]

# 2. 初始化OpenAI客户端
openai_client = OpenAI(
    api_key = "",
    base_url="https://api.chatanywhere.tech/v1"
)

def safe_html_parse(html_content):
    """
    安全解析HTML内容，如果解析失败则返回None
    尝试多种解析器提高兼容性[2,8](@ref)
    """
    if not html_content or len(html_content.strip()) == 0:
        logger.warning("HTML内容为空")
        return None
    
    # 尝试不同的解析器（按容错性排序）
    parsers = ['lxml', 'html.parser', 'html5lib']
    
    for parser in parsers:
        try:
            soup = BeautifulSoup(html_content, parser)
            # 简单验证解析是否成功（检查是否有基本的HTML结构）
            if soup.find() is not None:
                logger.debug(f"使用 {parser} 解析器成功解析HTML")
                return soup
        except Exception as e:
            logger.warning(f"解析器 {parser} 失败: {str(e)}")
            continue
    
    # 如果所有解析器都失败，尝试宽松解析模式
    try:
        # 使用html5lib，它通常具有最好的容错性[2](@ref)
        soup = BeautifulSoup(html_content, 'html5lib')
        logger.info("使用html5lib解析器（宽松模式）完成解析")
        return soup
    except Exception as e:
        logger.error(f"所有HTML解析尝试均失败: {str(e)}")
        return None

# 3. 定义文本向量化函数（使用OpenAI embeddings）
def vectorize_text(text, max_retries=3):
    """
    增强版文本向量化函数，解决API限制问题
    """
    # 检查文本是否为空
    if not text or len(text.strip()) == 0:
        logger.warning("文本为空，跳过向量化")
        return None
    
    # 限制文本长度（OpenAI API限制）
    MAX_TOKENS = 8000  # 保留安全边际
    if len(text) > MAX_TOKENS:
        logger.warning(f"文本过长({len(text)}字符)，截断前{MAX_TOKENS}字符")
        text = text[:MAX_TOKENS]
    
    # 指数退避重试机制
    for attempt in range(max_retries):
        try:
            response = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                dimensions=1024,
                encoding_format="float"
            )
            return response.data[0].embedding
        except RateLimitError:
            wait_time = (2 ** attempt) * 5  # 指数退避策略
            logger.warning(f"速率限制，等待 {wait_time}秒后重试... (尝试 {attempt+1}/{max_retries})")
            time.sleep(wait_time)
        except APIConnectionError:
            logger.warning("网络连接问题，10秒后重试...")
            time.sleep(10)
        except Exception as e:
            logger.error(f"向量生成失败: {e}")
            return None
    
    logger.error(f"达到最大重试次数 ({max_retries})，放弃处理")
    return None

# 4. 处理MongoDB中的每条文档（从指定ID开始，只处理未向量化的文档）
def process_documents_from_id(start_id):
    """
    从指定的_id开始处理未向量化的文档
    start_id: 起始文档的_id字符串，例如'68a00b57d7c172ed0c81ec5d'
    """
    # 将字符串ID转换为ObjectId对象
    try:
        start_object_id = ObjectId(start_id)
    except Exception as e:
        logger.error(f"提供的起始ID格式无效 - {e}")
        return

    # 查询条件：从指定ID开始，且没有'vector'字段的文档
    query = {
        '_id': {'$gte': start_object_id},
        'content': {'$exists': True},
        'vector': {'$exists': False}
    }
    
    # 按_id排序以确保顺序处理
    cursor = collection.find(query).sort('_id', 1)
    total_docs = collection.count_documents(query)
    processed_count = 0
    success_count = 0
    parse_failed_count = 0
    vector_failed_count = 0
    
    logger.info(f"找到 {total_docs} 个未向量化的文档需要处理（从ID {start_id} 开始）")
    
    if total_docs == 0:
        logger.info("没有需要处理的文档")
        return
    
    start_time = datetime.now()
    
    for doc in cursor:
        # 5. 使用增强的HTML解析函数[3,6](@ref)
        html_content = doc['content']
        
        # 安全解析HTML，如果失败则跳过当前文档
        soup = safe_html_parse(html_content)
        if soup is None:
            parse_failed_count += 1
            logger.warning(f"文档 {doc['_id']} HTML解析失败，已跳过 ({parse_failed_count}个解析失败)")
            # 可选：记录解析失败的文档ID以便后续检查
            collection.update_one(
                {'_id': doc['_id']},
                {'$set': {'parse_status': 'failed'}}
            )
            continue
        
        # 提取纯文本
        plain_text = soup.get_text(separator=' ', strip=True)
        
        # 检查提取的文本是否有效
        if not plain_text or len(plain_text.strip()) < 10:  # 假设有效文本至少10个字符
            logger.warning(f"文档 {doc['_id']} 提取的文本过短或为空，跳过向量化")
            parse_failed_count += 1
            continue
        
        # 6. 文本向量化（添加API调用间隔）
        time.sleep(1)  # 基础间隔
        vector = vectorize_text(plain_text)
        if vector is None:
            vector_failed_count += 1
            logger.warning(f"文档 {doc['_id']} 向量化失败，跳过 ({vector_failed_count}个向量化失败)")
            continue
        
        # 7. 更新文档，添加向量字段
        collection.update_one(
            {'_id': doc['_id']},
            {'$set': {'vector': vector, 'parse_status': 'success'}}
        )
        
        success_count += 1
        processed_count += 1
        logger.info(f"文档 {doc['_id']} 处理成功 ({success_count}/{total_docs})，向量维度: {len(vector)}")
        
        # 每处理10个文档显示一次进度
        if processed_count % 10 == 0:
            elapsed_time = datetime.now() - start_time
            docs_per_minute = success_count / (elapsed_time.total_seconds() / 60) if elapsed_time.total_seconds() > 0 else 0
            logger.info(f"进度: 成功{success_count}/失败{parse_failed_count+vector_failed_count}/总计{total_docs} | 速度: {docs_per_minute:.1f} 文档/分钟")
            
            # 批量处理后的额外休眠
            time.sleep(5)  # 每10个文档额外休息5秒
    
    # 最终统计报告
    end_time = datetime.now()
    total_time = end_time - start_time
    logger.info(f"处理完成！成功: {success_count}, 解析失败: {parse_failed_count}, 向量化失败: {vector_failed_count}")
    logger.info(f"总用时: {total_time}，平均速度: {success_count/max(1, total_time.total_seconds()/60):.1f} 文档/分钟")

# 执行处理
if __name__ == "__main__":
    logger.info("脚本开始执行...")
    
    # 指定起始ID
    start_id = "689c165b388b37bf997f758e"
    
    process_documents_from_id(start_id)
    logger.info("脚本执行完毕！")