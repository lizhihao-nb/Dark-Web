import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm  # <-- 新增导入

# 读取偏好度矩阵
df = pd.read_excel("lik.xlsx", sheet_name="Sheet1", index_col=0)

# 类别名称（可选：缩短以便显示）
short_names = [
    "Marketplaces",
    "Forums & Directories",
    "Financial Fraud",
    "Hacking",
    "Drug Trade",
    "Firearms",
    "Sexual Exploitation",
    "OPSEC Guidance",
    "Hosting Infrastructure",
    "Digital Preservation",
    "Law & Political Report"
]

# 创建热力图
plt.figure(figsize=(10, 8))

# ✅ 关键修改：使用 PowerNorm 增强低值对比度
# gamma=0.4 表示对低值区域进行拉伸（值越小，效果越强）
norm = PowerNorm(gamma=0.5, vmin=0, vmax=1)
im = plt.imshow(df.values, cmap='YlGnBu', aspect='auto', norm=norm)

# 设置坐标轴标签
plt.xticks(ticks=np.arange(len(short_names)), labels=short_names, rotation=45, ha='right')
plt.yticks(ticks=np.arange(len(short_names)), labels=short_names)

# 添加数值标签
for i in range(df.shape[0]):
    for j in range(df.shape[1]):
        val = df.iloc[i, j]
        # 文本颜色阈值也需调整（因为视觉上深色区域变多了）
        color = "white" if val > 0.3 else "black"  # 从 0.5 降到 0.3
        plt.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

# 美化
# plt.title("Connection Preference Matrix\n(Enhanced Low-Value Contrast)", fontsize=14, pad=20)
plt.colorbar(im, shrink=0.8, label='Preference Score')
plt.tight_layout()

# 保存并显示
plt.savefig("preference_heatmap_enhanced.png", dpi=300, bbox_inches='tight')
plt.show()