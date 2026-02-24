import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ── 全局設置 ───────────────────────────────────────────────────────────────
rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 10

# 文件路徑（使用 raw 字串，避免轉義問題）
FILE_PATH = r"F:\dataset\肿瘤微环境TME.txt"

# ── 解析 txt 文件 ────────────────────────────────────────────────────────
def parse_tme_txt(filepath):
    # 檢查檔案是否存在
    if not os.path.exists(filepath):
        print(f"【錯誤】檔案不存在：{filepath}")
        return {}

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        print(f"【讀檔失敗】{e}")
        return {}

    samples = {}
    current_sample = None
    current_section = None
    current_section_inner = None

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # 樣本名稱匹配（更寬鬆）
        if re.match(r'^[A-Za-z0-9\-\\]+：?$', line):
            current_sample = line.rstrip('：').strip()
            samples[current_sample] = {'h5': {}, 'json': {}}
            current_section = None
            current_section_inner = None
            continue

        # 相似度評分
        if '相似度评分' in line:
            score_match = re.search(r'(\d+\.\d+)', line)
            if score_match and current_sample:
                samples[current_sample]['similarity_score'] = float(score_match.group(1))
            continue

        # 進入 h5_results / json_results
        if 'h5_results' in line:
            current_section = 'h5'
            continue
        if 'json_results' in line:
            current_section = 'json'
            continue

        # counts 開始
        if line.startswith('"counts": {'):
            current_section_inner = 'counts'
            continue

        # counts 結束
        if current_section_inner == 'counts' and '}' in line:
            current_section_inner = None
            continue

        # 解析 counts 內的鍵值對
        if current_section_inner == 'counts' and ':' in line:
            try:
                k, v = line.split(':', 1)
                k = k.strip().strip('"')
                v = int(v.strip().rstrip(','))
                if current_section in samples[current_sample]:
                    samples[current_sample][current_section][f'counts_{k}'] = v
            except:
                pass
            continue

        # 普通鍵值對（浮點數）
        if current_section and ':' in line:
            try:
                k, v = line.split(':', 1)
                k = k.strip().lower().strip('"')
                v_str = v.strip().rstrip(',')
                try:
                    v = float(v_str)
                    if current_section in samples[current_sample]:
                        samples[current_sample][current_section][k] = v
                except ValueError:
                    pass  # 暫不處理列表類型
            except:
                pass

    print(f"解析完成，共找到 {len(samples)} 個樣本")
    print("樣本名稱列表：", list(samples.keys()))
    return samples


# ── 主程序 ────────────────────────────────────────────────────────────────
print("當前工作目錄：", os.getcwd())
print("目標檔案是否存在？", os.path.exists(FILE_PATH))

data = parse_tme_txt(FILE_PATH)

# 如果 data 為空，後續不會崩潰，但會有警告
if not data:
    print("警告：沒有解析到任何樣本，請檢查檔案內容格式")

# 樣本順序（注意反斜杠）
sample_order = [
    "ZHAOXINBAO", "ZHUNANXING", "ZHUZHULIN",
    r"X24-13356\A11", r"X24-13356\A12", "X24-13356-A13",
    "X24-13356-A14", "X24-13356-A15"
]

# 建構 DataFrame
rows = []
for s in sample_order:
    if s not in data:
        print(f"警告：樣本 {s} 未找到，跳過")
        continue
    h = data[s].get('h5', {})
    j = data[s].get('json', {})
    total_h5 = h.get('total_nuclei', 1) or 1
    total_json = j.get('total_nuclei', 1) or 1

    row = {
        'sample': s.replace('\\', '/'),
        'similarity': data[s].get('similarity_score', np.nan),
        'tumor_purity_h5': h.get('tumor_purity'),
        'tumor_purity_json': j.get('tumor_purity'),
        'stromal_h5': h.get('stromal_score'),
        'stromal_json': j.get('stromal_score'),
        'immune_h5': h.get('immune_score'),
        'immune_json': j.get('immune_score'),
        'neopla_h5': h.get('counts_neopla', 0) / total_h5,
        'neopla_json': j.get('counts_neopla', 0) / total_json,
        'inflam_h5': h.get('counts_inflam', 0) / total_h5,
        'inflam_json': j.get('counts_inflam', 0) / total_json,
        'connec_h5': h.get('counts_connec', 0) / total_h5,
        'connec_json': j.get('counts_connec', 0) / total_json,
    }
    rows.append(row)

df = pd.DataFrame(rows).set_index('sample')

print("\nDataFrame 前幾行：")
print(df.round(4))

# ── 主圖1：多指標分組條形圖 + 對角線散點 ────────────────────────────────
if not df.empty:
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
    plt.savefig('main_fig1.png', dpi=300, bbox_inches='tight')
    plt.show()

# ── 主圖3：改為三個並排餅圖（類似 d/e/f 風格） ────────────────────────────
if not df.empty:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))

    # 選擇三個代表性樣本的 neoplastic 比例做示範
    examples = [
        ("ZHAOXINBAO", "H5", df.loc["ZHAOXINBAO", "neopla_h5"] if "ZHAOXINBAO" in df.index else 0.2),
        ("X24-13356/A14", "JSON", df.loc["X24-13356/A14", "neopla_json"] if "X24-13356/A14" in df.index else 0.4),
        ("X24-13356-A15", "JSON", df.loc["X24-13356-A15", "neopla_json"] if "X24-13356-A15" in df.index else 0.6),
    ]

    colors_main = '#d32f2f'
    colors_sub  = '#bbdefb'

    for i, (sample_name, method, neopla_ratio) in enumerate(examples):
        ax = axes[i]
        values = [neopla_ratio, 1 - neopla_ratio]
        labels = ['neoplastic', 'others']

        wedges, _ = ax.pie(
            values,
            colors=[colors_main, colors_sub],
            startangle=90,
            counterclock=False,
            wedgeprops=dict(width=0.5, edgecolor='white')
        )

        # 中心顯示主要比例
        ax.text(0, 0, f"{neopla_ratio:.1%}", ha='center', va='center',
                fontsize=16, fontweight='bold', color='white')

        # 圖例
        ax.legend(wedges, labels, loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=9)

        # 標題
        ax.set_title(f"{sample_name}\n{method}", fontsize=11)

        ax.axis('equal')

    fig.suptitle("細胞類型比例示例（neoplastic vs others）", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, wspace=0.5)
    plt.savefig('main_fig3_pie.png', dpi=300, bbox_inches='tight')
    plt.show()

print("\n執行結束")
print("如果圖表沒有生成，請檢查上面是否有任何警告或錯誤訊息")
# ── 主圖3：三個並排餅圖（帶數字在上方 + 拉出一塊 + 詳細說明） ────────────────
if not df.empty:
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.2))

    # 定義三個圖的資料（你可以根據需要替換樣本或指標）
    pie_configs = [
        # d - 類似 100% 正確
        {
            "title": "d",
            "main_label": "PCa",
            "sub_label": "Non-PCa",
            "main_value": 1.00,
            "sub_value": 0.00,
            "n": 64,
            "accuracy": "100%",
            "overdiagnosis": "0%",
            "explode_idx": 0,          # 哪一塊要拉出來（0=主塊，1=副塊）
            "detail_pos": (0.4, -0.9), # 詳細文字的位置偏移
        },
        # e - 91.4% 正確 + 8.6% 過診斷
        {
            "title": "e",
            "main_label": "CSPCa",
            "sub_label": "Non-CSPCa",
            "main_value": 0.914,
            "sub_value": 0.086,
            "n": 70,
            "accuracy": "91.4%",
            "overdiagnosis": "8.6%",
            "explode_idx": 1,          # 拉出過診斷那塊
            "detail_pos": (0.4, -0.9),
        },
        # f - 52.2% 修正
        {
            "title": "f",
            "main_label": "Concordance",
            "sub_label": "Up- or downgrading",
            "main_value": 0.522,
            "sub_value": 0.478,
            "n": 23,
            "accuracy": "Correction: 52.2%",
            "overdiagnosis": "n=11 patients",
            "explode_idx": 1,          # 拉出不一致部分
            "detail_pos": (0.4, -1.0),
        }
    ]

    colors = ['#d32f2f', '#bbdefb']   # 紅 + 淺藍

    for i, cfg in enumerate(pie_configs):
        ax = axes[i]

        explode = [0, 0]
        explode[cfg["explode_idx"]] = 0.12   # 拉出來的幅度，可調 0.08~0.15

        values = [cfg["main_value"], cfg["sub_value"]]
        labels = [cfg["main_label"], cfg["sub_label"]]

        wedges, texts = ax.pie(
            values,
            explode=explode,
            labels=None,
            colors=colors,
            startangle=90,
            counterclock=False,
            wedgeprops=dict(width=0.52, edgecolor='white', linewidth=1.1),
            pctdistance=0.75
        )

        # 在中心顯示主要比例（大字）
        ax.text(0, 0.08, f"{cfg['main_value']:.1%}",
                ha='center', va='center', fontsize=17, fontweight='bold', color='white')

        # 上方顯示 n = xx patients
        ax.text(0, 1.15, f"n = {cfg['n']} patients",
                ha='center', va='bottom', fontsize=10, fontweight='semibold')

        # 圖例（放在右側）
        ax.legend(wedges, labels,
                  loc='center left', bbox_to_anchor=(1.05, 0.5),
                  fontsize=9.5, frameon=False)

        # 標題（d / e / f）
        ax.set_title(cfg["title"], fontsize=13, pad=20, fontweight='bold')

        # 下方詳細說明（Accuracy / Overdiagnosis / Correction）
        detail_text = f"Accuracy: {cfg['accuracy']}\nOverdiagnosis: {cfg['overdiagnosis']}"
        if "Correction" in cfg['accuracy']:
            detail_text = f"{cfg['accuracy']}\n{cfg['overdiagnosis']}"

        ax.text(*cfg["detail_pos"], detail_text,
                ha='center', va='top', fontsize=9.5,
                bbox=dict(facecolor='white', edgecolor='gray',
                          boxstyle="round,pad=0.4", alpha=0.95))

        # 底部共同說明文字
        if i == 1:  # 只在中間放一次
            ax.text(0, -1.35, "Prospective prediction by MRI-PTP Ca",
                    ha='center', va='bottom', fontsize=10, fontstyle='italic',
                    bbox=dict(facecolor='#f5f5f5', edgecolor='none', pad=5))

        ax.axis('equal')

    fig.suptitle("", fontsize=1)  # 避免多餘標題
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.12, wspace=0.55)
    plt.savefig('main_fig3_pie_detailed.png', dpi=350, bbox_inches='tight')
    plt.show()

else:
    print("DataFrame 為空，無法繪製餅圖")