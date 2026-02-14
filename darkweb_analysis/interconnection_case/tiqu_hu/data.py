import json
import re
from collections import defaultdict, deque

# 1. 读取 data.js 并提取 JSON 数据
def load_data_js(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # 去掉 "const data = " 和末尾分号/空格
    json_str = re.sub(r'^\s*const\s+data\s*=\s*', '', content)
    json_str = re.sub(r'\s*;\s*$', '', json_str)
    return json.loads(json_str)

# 2. 构建无向图并找连通分量
def find_connected_components(nodes, edges):
    graph = defaultdict(set)
    node_ids = {node['id'] for node in nodes}
    for node_id in node_ids:
        graph[node_id] = set()

    for edge in edges:
        src, tgt = edge['source'], edge['target']
        if src in graph and tgt in graph:
            graph[src].add(tgt)
            graph[tgt].add(src)

    visited = set()
    components = []

    for node_id in graph:
        if node_id not in visited:
            component = []
            queue = deque([node_id])
            visited.add(node_id)
            while queue:
                current = queue.popleft()
                component.append(current)
                for neighbor in graph[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(component)
    return components

# 3. 主函数：提取最多 5 个符合条件的子图
def main():
    input_file = 'data.js'
    output_prefix = 'subgraph'  # 输出文件前缀
    min_size = 20
    max_size = 300
    max_output = 50  # 最多输出 5 个子图

    print("🔍 正在加载 data.js...")
    data = load_data_js(input_file)
    nodes = data['nodes']
    edges = data['edges']

    print(f"✅ 加载完成：{len(nodes)} 节点, {len(edges)} 边")

    label_map = {node['id']: node['label'] for node in nodes}

    print("🔍 正在计算连通分量...")
    components = find_connected_components(nodes, edges)
    print(f"✅ 共找到 {len(components)} 个连通分量")

    selected_components = []
    for comp in components:
        if not (min_size <= len(comp) <= max_size):
            continue
        labels = {label_map.get(nid, '') for nid in comp}
        if 'marketNode' in labels and 'financeNode' in labels:
            selected_components.append(comp)
            if len(selected_components) >= max_output:
                break  # 找够 5 个就停止

    if not selected_components:
        print(f"❌ 未找到同时包含 market 和 finance、且大小在 [{min_size}, {max_size}] 的连通子图")
        return

    print(f"✅ 找到 {len(selected_components)} 个符合条件的子图（最多输出 {max_output} 个）")

    for idx, comp in enumerate(selected_components, start=1):
        selected_set = set(comp)
        sub_nodes = [node for node in nodes if node['id'] in selected_set]
        sub_edges = [
            edge for edge in edges
            if edge['source'] in selected_set and edge['target'] in selected_set
        ]

        # 构造 JS 格式内容：const subgraph = {...};
        subgraph_dict = {'nodes': sub_nodes, 'edges': sub_edges}
        json_str = json.dumps(subgraph_dict, indent=2, ensure_ascii=False)
        js_content = f"const subgraph = {json_str};\n"

        output_file = f"{output_prefix}_{idx}.js"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(js_content)

        market_count = sum(1 for n in sub_nodes if n['label'] == 'marketNode')
        finance_count = sum(1 for n in sub_nodes if n['label'] == 'financeNode')
        print(f"✅ 已保存子图 {idx} 到 {output_file} "
              f"({len(sub_nodes)} 节点, {len(sub_edges)} 边) | "
              f"Market: {market_count}, Finance: {finance_count}")

if __name__ == '__main__':
    main()