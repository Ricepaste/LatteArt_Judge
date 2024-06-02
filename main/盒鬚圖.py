import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 定義期望值和標準差
# data = {
#     'Elo': (1.377, 0.183),
#     'Random Walk': (0.410, 0.15),
#     'Bradley-Terry': (0.295, 0.16),
#     'Thurstone': (0.000, 0.0)
# }
data = {
    'Elo': (41.36, 32.8),
    'Random Walk': (30.79, 11.06),
    'Bradley-Terry': (29.36, 10.12),
    'Thurstone': (26.5, 8.87)
}

# 計算四分位數和鬚端
result = {}
for key, (mean, std) in data.items():
    min_val = mean - 2.698 * std
    q1 = mean - 0.675 * std
    median = mean
    q3 = mean + 0.675 * std
    max_val = mean + 2.698 * std

    result[key] = [min_val, q1, median, q3, max_val]

# 轉換為盒鬚圖所需格式
labels = []
values = []
colors = ['blue', 'red', 'green', 'yellow']  # 對應的顏色
for label, val in result.items():
    labels.append(label)
    values.append(val)

# 繪製盒鬚圖
fig, ax = plt.subplots()
box = ax.boxplot(values, vert=False, patch_artist=True)
ax.set_yticklabels(labels, fontsize=16)  # 增大字體大小
ax.set_title(
    'Approximate Box Plot using Mean and Standard Deviation', fontsize=16)
ax.set_xlabel('Values', fontsize=16)

# 設置顏色
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)  # type: ignore

plt.show()
