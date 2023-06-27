import json
import matplotlib.pyplot as plt
import numpy as np

# 读取t_result.json文件
with open('t_result.json', 'r') as file:
    data = json.load(file)

# 从字典中提取数据
non_accumulated_result = data.get('non_accumulated_result', [])
accumulated_result = data.get('accumulated_result', [])

# 创建新图形
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# 在第一个子图中绘制非累加结果
axs[0].scatter(range(len(non_accumulated_result)), non_accumulated_result, label="Non Accumulated Result")
axs[0].set_title('Non Accumulated Result')
axs[0].set_xlabel('Index')
axs[0].set_ylabel('Time')
axs[0].legend()

# 在第二个子图中绘制累加结果
axs[1].plot(accumulated_result, label="Accumulated Result", marker='x')
axs[1].set_title('Accumulated Result')
axs[1].set_xlabel('Index')
axs[1].set_ylabel('Time')
axs[1].legend()

# 调整子图的间距
plt.tight_layout()

# 显示图形
plt.show()

# 计算累加结果的斜率
if accumulated_result:
    indexes = np.arange(len(accumulated_result))
    slope, intercept = np.polyfit(indexes, accumulated_result, 1)
    print(f"The slope of the accumulated result is: {slope}")
else:
    print("No accumulated_result data found.")
