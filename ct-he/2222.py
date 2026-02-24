# -*- coding: utf-8 -*-
"""
修复版：从完整 txt 解析 TME 数据并绘制多面板对比图
路径: 假设文件在 "F:\\dataset\\肿瘤微环境TME.txt" 或你自己的路径
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from adjustText import adjust_text  # pip install adjustText （可选，防标签重叠）

# ────────────────────────────────────────────────
# 1. 读取并解析 txt 文件（更鲁棒版本）
# ────────────────────────────────────────────────
file_path = r"F:\dataset\肿瘤微环境TME.txt"  # ← 修改成你的实际路径

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# 分割每个样本块（以样本名： + ====== 开头）
blocks = re.split(r'(\w+(?:-\\|[-A-Za-z0-9\\])+)：\n={10,}', text)

data_list = []

current_sample = None
current_section = None

lines = text.splitlines()
i = 0
while i < len(lines):
    line = lines[i].strip()

    # 样本名检测（允许反斜杠、-Axx 后缀）
    sample_match = re.match(r'^([^\s：]+)：$', line)
    if sample_match:
        current_sample = sample_match.group(1).replace('\\', '/')
        data_list.append({'sample': current_sample, 'similarity': np.nan})
        i += 1
        continue

    # 相似度
    if '相似度评分】:' in line and current_sample:
        try:
            score_str = re.search(r':\s*([\d.]+)\s*\(', line).group(1)
            for d in data_list:
                if d['sample'] == current_sample:
                    d['similarity'] = float(score_str)
        except:
            pass
        i += 1
        continue

    # 进入 h5_results 或 json_results
    if '"h5_results":' in line:
        current_section = 'h5'
    elif '"json_results":' in line:
        current_section = 'json'

    # 提取关键数值行
    if current_sample and current_section and ':' in line and not line.startswith('"avg_ripley_k"'):
        try:
            key, val_str = [part.strip() for part in line.split(':', 1)]
            key = key.strip('" ,')
            val_str = val_str.rstrip(',').strip()

            if val_str.replace('.', '').replace('-', '').isdigit() or 'e' in val_str:
                val = float(val_str)

                for d in data_list:
                    if d['sample'] == current_sample:
                        d[f'{key}_{current_section}'] = val
        except:
            pass

    i += 1

# 转成 DataFrame，只保留我们关心的列
df = pd.DataFrame(data_list)
df = df[['sample', 'similarity',
         'tumor_purity_h5', 'tumor_purity_json',
         'stromal_score_h5', 'stromal_score_json',
         'immune_score_h5', 'immune_score_json']]

# 重命名列，便于绘图
df = df.rename(columns={
    'stromal_score_h5': 'stromal_h5',
    'stromal_score_json': 'stromal_json',
    'immune_score_h5': 'immune_h5',
    'immune_score_json': 'immune_json'
})

print("\n解析结果预览（应有 8 行）：")
print(df)
print("\n样本数:", len(df))

if len(df) < 1:
    print("解析失败！请检查 txt 路径或格式。")
    exit()

# ────────────────────────────────────────────────
# 2. 绘图 - 多面板（类似论文风格）
# ────────────────────────────────────────────────
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(20, 14))
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.4)

# A: 相似度水平条形图（从高到低排序）
ax_a = fig.add_subplot(gs[0, :2])
df_sorted = df.sort_values('similarity', ascending=True)
ax_a.barh(df_sorted['sample'], df_sorted['similarity'], color='cornflowerblue')
ax_a.set_xlim(50, 100)
ax_a.set_xlabel('相似度 (%)')
ax_a.set_title('A. H5 vs JSON TME 结果相似度', fontsize=14)
ax_a.grid(axis='x', ls='--', alpha=0.5)

# B: Tumor purity 散点对比 + 标签
ax_b = fig.add_subplot(gs[0, 2:])
ax_b.scatter(df['tumor_purity_h5'], df['tumor_purity_json'], s=100, c='darkred', edgecolor='white', zorder=3)
ax_b.plot([0, 1], [0, 1], 'k--', lw=1.5, label='y = x')
texts = []
for x, y, s in zip(df['tumor_purity_h5'], df['tumor_purity_json'], df['sample']):
    texts.append(ax_b.text(x, y, s, fontsize=9, ha='left', va='bottom'))
adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.7))
ax_b.set_xlabel('H5 tumor purity')
ax_b.set_ylabel('JSON tumor purity')
ax_b.set_title('B. 肿瘤纯度对比', fontsize=14)
ax_b.legend()
ax_b.set_aspect('equal', adjustable='box')

# C: 堆叠柱状 - 只画 H5 的细胞比例（counts 需要额外解析，此处简化用 purity + non-tumor）
# 如需完整比例，可再扩展解析 counts 部分
ax_c = fig.add_subplot(gs[1, :])
width = 0.4
x = np.arange(len(df))
ax_c.bar(x - width / 2, df['tumor_purity_h5'], width, label='H5 Tumor', color='#e74c3c')
ax_c.bar(x + width / 2, df['tumor_purity_json'], width, label='JSON Tumor', color='#3498db')
ax_c.set_xticks(x)
ax_c.set_xticklabels(df['sample'], rotation=45, ha='right')
ax_c.set_ylabel('Tumor Purity')
ax_c.set_title('C. H5 vs JSON 肿瘤纯度柱状对比', fontsize=14)
ax_c.legend()

# D: Stromal vs Immune 散点（两种方法叠加）
ax_d = fig.add_subplot(gs[2, :2])
ax_d.scatter(df['stromal_h5'], df['immune_h5'], s=80, c='forestgreen', label='H5', alpha=0.8)
ax_d.scatter(df['stromal_json'], df['immune_json'], s=80, c='orange', marker='^', label='JSON', alpha=0.8)
texts = [ax_d.text(x + 0.005, y, s, fontsize=8) for x, y, s in zip(df['stromal_h5'], df['immune_h5'], df['sample'])]
adjust_text(texts)
ax_d.set_xlabel('Stromal Score')
ax_d.set_ylabel('Immune Score')
ax_d.set_title('D. 基质 vs 免疫分数对比', fontsize=14)
ax_d.legend()

# E: 简单相关性总结（文本 + 小图）
ax_e = fig.add_subplot(gs[2, 2:])
ax_e.axis('off')
stats_text = f"平均相似度: {df['similarity'].mean():.2f}%\n" \
             f"肿瘤纯度平均偏差: {abs(df['tumor_purity_h5'] - df['tumor_purity_json']).mean():.4f}\n" \
             f"样本数: {len(df)}"
ax_e.text(0.1, 0.5, stats_text, fontsize=12, va='center')

plt.suptitle("肿瘤微环境 TME 分析 - H5 与 JSON 方法对比", fontsize=18, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])

# plt.savefig("tme_comparison_fixed.png", dpi=300, bbox_inches='tight')
plt.show()