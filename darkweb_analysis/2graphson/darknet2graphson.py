graphson_dir = "./darknet_graphson/"

import pymongo
import json
import time
import os
import re

# 连接到 MongoDB
client = pymongo.MongoClient("mongodb://192.168.31.9:27017/")
db = client['3our_spider_db']
collection = db["filtered_all"]

# 确保输出目录存在
os.makedirs(graphson_dir, exist_ok=True)

# 洋葱地址正则（56位）
ONION_PATTERN = re.compile(r'([a-z0-9]{56})\.onion')

def extract_onion_address(url):
    """从 URL 中提取 56 位洋葱地址，若无则返回 None"""
    match = ONION_PATTERN.search(url)
    return match.group(1) if match else None

def add_inE(inE_dict, outV, edge, ID):
    inE_list = inE_dict.get("flow", [])
    tmp_dict = {"id": ID, "outV": outV, "properties": {"edge_kind": edge}}
    inE_list.append(tmp_dict)
    inE_dict["flow"] = inE_list
    return inE_dict

def add_outE(outE_dict, inV, edge, ID):
    outE_list = outE_dict.get("flow", [])
    tmp_dict = {"id": ID, "inV": inV, "properties": {"edge_kind": edge}}
    outE_list.append(tmp_dict)
    outE_dict["flow"] = outE_list
    return outE_dict

def add_properties(properties_dict, value, level):
    properties_dict.update(
        value=[{"id": "test_version", "value": value}],
        level=[{"id": "test_version", "value": level}]
    )
    return properties_dict

def darkweb_to_graphson(data_list, file_num):
    """处理任意长度的数据列表（包括少于 BATCH_SIZE 的情况）"""
    if not data_list:
        print("数据为空，跳过写入。")
        return False

    file_links_dict = {}

    for link in data_list:
        id_str = str(link["_id"])
        url1 = link["url1"]
        url2 = link["url2"]
        edge_type = link['edge']

        level_value1 = "site" if (url1.count('/') == 2 and url1.count('.') == 1) else "page"
        level_value2 = "site" if (str(url2).count('/') == 2 and str(url2).count('.') == 1) else "page"

        # --- 处理子节点 url2 ---
        if url2 not in ["-1", "1"]:
            link_url = url2
            if link_url in file_links_dict:
                node = file_links_dict[link_url]
                inE_dict = node.get("inE", {})
                node["inE"] = add_inE(inE_dict, url1, edge_type, id_str)
            else:
                props = add_properties({}, "", level_value2)
                if link["type2"] == "dark":
                    onion = extract_onion_address(link_url)
                    if onion:
                        props["site"] = [{"id": "onion_address", "value": onion}]
                node = {
                    "id": link_url,
                    "label": link["type2"],
                    "properties": props,
                    "inE": add_inE({}, url1, edge_type, id_str)
                }
                file_links_dict[link_url] = node

        # --- 处理父节点 url1 ---
        if url1 != "NULL":
            link_url = url1
            if link_url in file_links_dict:
                node = file_links_dict[link_url]
                value_flag = "0" if url2 == "-1" else "1"
                node["properties"] = add_properties(node.get("properties", {}), value_flag, level_value1)
                if node["label"] == "dark" and "site" not in node["properties"]:
                    onion = extract_onion_address(link_url)
                    if onion:
                        node["properties"]["site"] = [{"id": "onion_address", "value": onion}]
                if url2 not in ["-1", "1"]:
                    outE_dict = node.get("outE", {})
                    node["outE"] = add_outE(outE_dict, url2, edge_type, id_str)
            else:
                value_flag = "0" if url2 == "-1" else "1"
                props = add_properties({}, value_flag, level_value1)
                if link["type1"] == "dark":
                    onion = extract_onion_address(link_url)
                    if onion:
                        props["site"] = [{"id": "onion_address", "value": onion}]
                node = {"id": link_url, "label": link["type1"], "properties": props}
                if url2 not in ["-1", "1"]:
                    node["outE"] = add_outE({}, url2, edge_type, id_str)
                file_links_dict[link_url] = node

    # 写入文件
    lines = [json.dumps(node, ensure_ascii=False) for node in file_links_dict.values()]
    current_json_file = f"Dark{file_num:05d}.json"
    with open(os.path.join(graphson_dir, current_json_file), 'w', encoding='utf-8') as fd:
        fd.write("\n".join(lines))

    print(f"✅ 成功写入 {len(lines)} 个节点到 {current_json_file}")
    return True

def monitor_and_process(batch_size=800000):
    """只要有未处理数据（>0），就返回最多 batch_size 条"""
    while True:
        count = collection.count_documents({"processed": {"$ne": True}})
        print(f"未处理文档数: {count}")
        if count > 0:
            documents = list(collection.find({"processed": {"$ne": True}}).limit(batch_size))
            return documents
        print("暂无数据，等待60秒...")
        time.sleep(60)

if __name__ == "__main__":
    count = 0
    BATCH_SIZE = 800000

    print("启动 GraphSON 导出服务...")
    while True:
        print("\n🔍 检查未处理数据...")
        test_list = monitor_and_process(BATCH_SIZE)

        print(f"开始处理第 {count} 批，共 {len(test_list)} 条记录")

        success = darkweb_to_graphson(test_list, count)

        if success:
            ids_to_mark = [doc["_id"] for doc in test_list]
            result = collection.update_many(
                {"_id": {"$in": ids_to_mark}},
                {"$set": {"processed": True}}
            )
            print(f"📌 标记 {result.modified_count} 条记录为已处理")
            print(f"✅ No.{count} graphson OK")
            count += 1
        else:
            print("❌ 写入失败！跳过本次批次，5分钟后重试...")
            time.sleep(300)

        print("当前批次处理完成。\n")