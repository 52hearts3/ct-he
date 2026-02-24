import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 顶刊风格设置（调整字体大小）
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
rcParams['axes.linewidth'] = 0.8
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['axes.labelsize'] = 10
rcParams['axes.titlesize'] = 11

# 数据
N = 10
theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
predicted = np.array([5407956, 206113, 4864, 408, 78, 20, 5, 2, 0, 2])
real = np.array([2567419, 302756, 17460, 1653, 242, 51, 16, 5, 2, 1])

# 归一化（log1p + 偏移避免零）
pred_norm = np.log1p(predicted) / np.log1p(predicted).max() + 0.12
real_norm = np.log1p(real) / np.log1p(real).max() + 0.12

width = (2 * np.pi) / N * 0.82  # 稍窄一点，给标签留空间

# 主图
fig = plt.figure(figsize=(8, 8), dpi=300)
ax = fig.add_subplot(111, polar=True)

# 内圈：预测
ax.bar(theta, pred_norm, width=width, bottom=0.0,
       color=plt.cm.Reds(pred_norm), alpha=0.78, edgecolor='white', linewidth=1.0)

# 外圈：真实
ax.bar(theta + width*0.04, real_norm, width=width*0.92, bottom=pred_norm + 0.04,
       color=plt.cm.Blues(real_norm), alpha=0.68, edgecolor='white', linewidth=0.8)

# bin 标签（向外推更远）
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_xticks(theta)
ax.set_xticklabels([f'Bin {i+1}' for i in range(N)], fontsize=9)
ax.tick_params(pad=18)  # 标签向外移更多

# 移除径向刻度
ax.set_yticks([])
ax.spines['polar'].set_visible(False)

# 中心文字（缩小并下移）
plt.text(0, -0.15, 'Nuclear Area\nDistribution\n(Pred vs Real)',
         ha='center', va='center', fontsize=10, fontweight='bold', color='0.4')

# 图例文字（放到右上角外部）
fig.text(0.82, 0.88, 'Prediction\n(inner, red scale)',
         color='#b2182b', fontsize=9, ha='left')
fig.text(0.82, 0.82, 'Ground truth\n(outer, blue scale)',
         color='#2166ac', fontsize=9, ha='left')

plt.tight_layout(pad=1.5)
plt.show()