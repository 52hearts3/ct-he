import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ── 设置（中文支持，可根据需要调整） ────────────────────────────────
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

# 文件路径（请确认路径正确）
FILE_PATH = r"F:\dataset\肿瘤微环境TME.txt"

# ── 解析 txt 文件 ───────────────────────────────────────────────
def parse_tme_txt(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    samples = {}
    current_sample = None
    current_section = None
    current_section_inner = None   # 关键修复：提前初始化，避免 UnboundLocalError

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # 样本名（匹配 ZHAOXINBAO、X24-13356\A11 等）
        if re.match(r'^[A-Za-z0-9\-\\]+：?$|^X24-13356[-A-Za-z0-9\\]+$', line.rstrip('：')):
            current_sample = line.rstrip('：').strip()
            samples[current_sample] = {}
            current_section = None
            current_section_inner = None
            continue

        # 相似度评分
        if '相似度评分' in line:
            score_match = re.search(r'(\d+\.\d+)', line)
            if score_match and current_sample:
                samples[current_sample]['similarity_score'] = float(score_match.group(1))
            continue

        # 进入 h5_results 或 json_results
        if 'h5_results' in line:
            current_section = 'h5'
            current_section_inner = None
            continue
        if 'json_results' in line:
            current_section = 'json'
            current_section_inner = None
            continue

        # 处理 counts 块开始
        if line.startswith('"counts": {'):
            current_section_inner = 'counts'
            continue

        # counts 块结束
        if current_section_inner == 'counts' and '}' in line:
            current_section_inner = None
            continue

        # counts 内部的键值对
        if current_section_inner == 'counts' and ':' in line:
            try:
                key_part, val_part = line.split(':', 1)
                key = key_part.strip().strip('"')
                val = val_part.strip().rstrip(',').strip()
                val = int(val)
                if current_sample and current_section:
                    key_name = f'counts_{key}'
                    if current_section not in samples[current_sample]:
                        samples[current_sample][current_section] = {}
                    samples[current_sample][current_section][key_name] = val
            except:
                pass
            continue

        # 普通键值对（tumor_purity, stromal_score 等）
        if current_section and ':' in line and current_sample:
            try:
                key_part, val_part = line.split(':', 1)
                key = key_part.strip().lower().strip('"')
                val = val_part.strip().rstrip(',').strip()
                try:
                    val = float(val)
                except ValueError:
                    pass
                if current_section not in samples[current_sample]:
                    samples[current_sample][current_section] = {}
                samples[current_sample][current_section][key] = val
            except:
                pass

    return samples


# ── 主程序 ────────────────────────────────────────────────────────────────
data = parse_tme_txt(FILE_PATH)

# 样本顺序（建议按相似度从高到低，或按文件出现顺序）
sample_order = [
    "ZHAOXINBAO", "ZHUNANXING", "ZHUZHULIN",
    "X24-13356\\A11", "X24-13356\\A12", "X24-13356-A13",
    "X24-13356-A14", "X24-13356-A15"
]

# 构建 DataFrame
rows = []
for s in sample_order:
    if s not in data:
        continue
    h = data[s].get('h5', {})
    j = data[s].get('json', {})
    row = {
        'sample': s,
        'similarity': data[s].get('similarity_score', np.nan),
        'tumor_purity_h5': h.get('tumor_purity'),
        'tumor_purity_json': j.get('tumor_purity'),
        'stromal_h5': h.get('stromal_score'),
        'stromal_json': j.get('stromal_score'),
        'immune_h5': h.get('immune_score'),
        'immune_json': j.get('immune_score'),
        'neopla_h5': h.get('counts_neopla', 0) / h.get('total_nuclei', 1) if h.get('total_nuclei') else 0,
        'neopla_json': j.get('counts_neopla', 0) / j.get('total_nuclei', 1) if j.get('total_nuclei') else 0,
        'inflam_h5': h.get('counts_inflam', 0) / h.get('total_nuclei', 1) if h.get('total_nuclei') else 0,
        'inflam_json': j.get('counts_inflam', 0) / j.get('total_nuclei', 1) if j.get('total_nuclei') else 0,
        'connec_h5': h.get('counts_connec', 0) / h.get('total_nuclei', 1) if h.get('total_nuclei') else 0,
        'connec_json': j.get('counts_connec', 0) / j.get('total_nuclei', 1) if j.get('total_nuclei') else 0,
    }
    rows.append(row)

df = pd.DataFrame(rows).set_index('sample')

# 打印解析结果检查（调试用）
print("解析得到的样本：", list(data.keys()))
print("\nDataFrame 前几行：")
print(df.head().round(4))

# ── 主图1：多指标分组条形图 + 对角线散点 ────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [3, 2]})

metrics = ['tumor_purity', 'stromal', 'immune']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
width = 0.25

x = np.arange(len(df))
for i, m in enumerate(metrics):
    ax1.bar(x - width + i*width, df[f'{m}_h5'], width, label=f'{m} H5', color=colors[i], alpha=0.85)
    ax1.bar(x + i*width, df[f'{m}_json'], width, label=f'{m} JSON', color=colors[i], alpha=0.45, hatch='//')

ax1.set_xticks(x)
ax1.set_xticklabels(df.index, rotation=45, ha='right', fontsize=9)
ax1.set_ylabel('Score')
ax1.set_title('Key TME Scores: H5 vs JSON')
ax1.legend(ncol=2, fontsize=9)
ax1.grid(True, axis='y', alpha=0.3)

# 对角线散点
for m, c in zip(metrics, colors):
    ax2.scatter(df[f'{m}_h5'], df[f'{m}_json'], s=60, label=m, alpha=0.9, color=c)
ax2.plot([0,1], [0,1], 'k--', lw=1.5, alpha=0.6)
ax2.set_xlabel('H5 value')
ax2.set_ylabel('JSON value')
ax2.set_title('H5 vs JSON (identity line)')
ax2.legend()
ax2.set_xlim(0, 0.8)
ax2.set_ylim(0, 0.8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('main_fig1_multi_bar_scatter.png', dpi=300, bbox_inches='tight')
plt.show()

# ── 主图2：Bland-Altman (tumor purity) ────────────────────────────────
mean = (df['tumor_purity_h5'] + df['tumor_purity_json']) / 2
diff = df['tumor_purity_h5'] - df['tumor_purity_json']
bias = diff.mean()
sd = diff.std()
loa_u = bias + 1.96 * sd
loa_l = bias - 1.96 * sd

plt.figure(figsize=(8, 6))
plt.scatter(mean, diff, s=80, c='royalblue', edgecolor='navy', alpha=0.9)
for i, txt in enumerate(df.index):
    plt.annotate(txt[:12], (mean.iloc[i], diff.iloc[i]), fontsize=9, xytext=(6,6), textcoords='offset points')
plt.axhline(0, color='gray', ls='--', lw=1)
plt.axhline(bias, color='crimson', ls='-', lw=2, label=f'Bias = {bias:.4f}')
plt.axhline(loa_u, color='orange', ls='--', lw=1.8, label=f'+1.96SD = {loa_u:.4f}')
plt.axhline(loa_l, color='orange', ls='--', lw=1.8, label=f'-1.96SD = {loa_l:.4f}')
plt.title('Bland–Altman Plot: Tumor Purity (H5 - JSON)')
plt.xlabel('Mean of H5 and JSON')
plt.ylabel('Difference')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('main_fig2_bland_altman.png', dpi=300, bbox_inches='tight')
plt.show()

# ── 主图3：细胞类型比例堆叠柱状图 ────────────────────────────────────────
cell_types = ['neopla', 'inflam', 'connec']
h5_cols = [f'{t}_h5' for t in cell_types]
json_cols = [f'{t}_json' for t in cell_types]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
df[h5_cols].plot(kind='bar', stacked=True, ax=ax1, colormap='tab20c', width=0.8)
df[json_cols].plot(kind='bar', stacked=True, ax=ax2, colormap='tab20c', width=0.8)

ax1.set_title('H5 - Cell Type Proportion')
ax2.set_title('JSON - Cell Type Proportion')
for ax in [ax1, ax2]:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Proportion')
    ax.set_ylim(0, 1)
    ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('main_fig3_stacked_bar.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n所有主图已保存到当前工作目录。")
print("如果需要 Ripley’s K 图，请提供 avg_ripley_k 列表的解析需求，我可以继续补充。")