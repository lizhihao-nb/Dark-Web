import pymongo
from scipy.spatial import distance
import numpy as np
from bson.objectid import ObjectId  # 关键：导入ObjectId

# 配置数据库连接信息
mongo_host = "mongodb://192.168.31.9:27017/"  # 请替换为您的实际IP和端口
db_name = "3our_spider_db"
collection_name = "content"
document_id_a = "689c16c1f9899cc545f2a416"  # 请替换为实际的ID，例如ObjectId字符串或对应值
document_id_b = "689c16cfc30c5e1da7018f76"  # 请替换为实际的ID

try:
    # 1. 连接到MongoDB数据库
    client = pymongo.MongoClient(mongo_host)
    db = client[db_name]
    collection = db[collection_name]
    print("数据库连接成功")

    # 2. 根据ID查询两个文档
    # 注意：您的文档ID字段名可能是'_id'或其他自定义字段，请根据实际情况修改查询条件
    doc_a = collection.find_one({"_id": ObjectId(document_id_a)})
    doc_b = collection.find_one({"_id": ObjectId(document_id_b)})

    if doc_a is None or doc_b is None:
        # 处理未找到文档的情况
        missing_ids = []
        if doc_a is None:
            missing_ids.append(str(document_id_a))
        if doc_b is None:
            missing_ids.append(str(document_id_b))
        raise ValueError(f"未找到ID为 {', '.join(missing_ids)} 的文档。请检查ID是否正确。")

    # 3. 提取文档中的vector字段
    vector_a = doc_a.get("vector")
    vector_b = doc_b.get("vector")

    if vector_a is None or vector_b is None:
        # 处理文档中不存在vector字段的情况
        missing_vectors = []
        if vector_a is None:
            missing_vectors.append("文档A")
        if vector_b is None:
            missing_vectors.append("文档B")
        raise ValueError(f"{' 和 '.join(missing_vectors)} 中未找到'vector'字段。")

    # 4. 检查向量数据格式并转换为NumPy数组
    # 确保向量是平坦的一维数组（形状应为(n,)
    # 如果您的向量存储为嵌套列表（如[[1,2,3]]），可能需要展平
    array_a = np.array(vector_a).flatten()
    array_b = np.array(vector_b).flatten()

    print(f"向量A的维度: {array_a.shape}")
    print(f"向量B的维度: {array_b.shape}")

    # 检查两个向量的维度是否相同
    if array_a.shape != array_b.shape:
        raise ValueError(f"向量维度不匹配。向量A: {array_a.shape}, 向量B: {array_b.shape}")

    # 5. 计算余弦距离
    # scipy的distance.cosine直接计算余弦距离，结果为 1 - 余弦相似度
    cosine_dist = distance.cosine(array_a, array_b)
    print("余弦距离:", cosine_dist)

except pymongo.errors.ConnectionFailure:
    print("错误: 无法连接到MongoDB数据库，请检查主机地址和端口。")
except ValueError as ve:
    print(f"数据错误: {ve}")
except Exception as e:
    print(f"发生未知错误: {e}")
finally:
    # 关闭数据库连接
    if 'client' in locals():
        client.close()
        print("数据库连接已关闭")