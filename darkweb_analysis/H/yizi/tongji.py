import pandas as pd
import numpy as np
import glob
import os

# ====== 配置目录路径 ======
data_dir = "/public/home/blockchain_2/slave1/darkanalysis/analysis/yizi/"  # ← 按需修改
# =========================

# 定义异质性区间
bins = [0.0, 0.5, 1.0, 2.0, 5.0, np.inf]
labels = ['[0.0, 0.5)', '[0.5, 1.0)', '[1.0, 2.0)', '[2.0, 5.0)', '≥5.0']

# 查找所有 CSV 文件
csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
print(f"📁 共找到 {len(csv_files)} 个 CSV 文件：")
for f in csv_files:
    print(f"  - {os.path.basename(f)}")

# 汇总统计结果
all_results = []

for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file)
        
        # 检查必要列是否存在
        if 'Heterogeneity' not in df.columns or 'NodeCount' not in df.columns:
            print(f"⚠️  跳过 {os.path.basename(csv_file)}：缺少 Heterogeneity 或 NodeCount 列")
            continue
        
        # 丢弃缺失值
        valid_df = df[['Heterogeneity', 'NodeCount']].dropna()
        total_components = len(valid_df)
        total_nodes = valid_df['NodeCount'].sum()
        
        if total_components == 0:
            print(f"⚠️  跳过 {os.path.basename(csv_file)}：无有效 Heterogeneity/NodeCount 数据")
            continue
        
        # 按异质性分箱
        valid_df['H_bin'] = pd.cut(valid_df['Heterogeneity'], bins=bins, labels=labels, right=False)
        
        # 按区间统计：分量数量 + 节点总数
        group_stats = valid_df.groupby('H_bin').agg(
            component_count=('Heterogeneity', 'count'),
            total_node_count=('NodeCount', 'sum')
        ).reindex(labels, fill_value=0)  # 确保所有区间都存在
        
        # 构建结果行
        row = {'Filename': os.path.basename(csv_file)}
        for label in labels:
            comp_count = group_stats.loc[label, 'component_count']
            node_sum = int(group_stats.loc[label, 'total_node_count'])  # 转为整数更易读
            comp_pct = round(comp_count / total_components * 100, 2) if total_components > 0 else 0.0
            
            row[f"{label}_comp%"] = comp_pct
            row[f"{label}_nodes"] = node_sum
        
        all_results.append(row)
        print(f"✅ {os.path.basename(csv_file)}: {total_components} 个分量, 总节点数 {int(total_nodes)}")
    
    except Exception as e:
        print(f"❌ 处理 {os.path.basename(csv_file)} 时出错: {e}")

# 生成汇总表格
if all_results:
    summary_df = pd.DataFrame(all_results)
    
    # 保存汇总 CSV
    output_file = os.path.join(data_dir, "H_distribution_summary_with_nodes.csv")
    summary_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ 汇总完成！结果已保存至:\n   {output_file}")
    print("\n📊 汇总预览（前5行）:")
    print(summary_df.head().to_string(index=False))
    
    # 可选：打印列名结构说明
    print("\n📌 输出列说明:")
    print("   - Filename: CSV 文件名")
    print("   - [区间]_comp%: 该异质性区间内连通分量数量占总分量数的百分比")
    print("   - [区间]_nodes: 该异质性区间内所有连通分量的节点总数（绝对值）")
else:
    print("❌ 未成功处理任何文件")