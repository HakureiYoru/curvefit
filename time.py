import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# 读取t_result.json文件
with open('t_result.json', 'r') as file:
    data = json.load(file)

# 从字典中提取数据
non_accumulated_result = data.get('non_accumulated_result', [])

# 创建图形
fig, ax = plt.subplots(figsize=(10, 5))

# 设置位移阈值
shift_threshold = np.pi

# 恢复位移的点
restored_result = np.array(non_accumulated_result)

# 查找位移超过阈值的点
shifted_indices = np.where(np.abs(np.diff(restored_result)) > shift_threshold)[0]

# 对超过阈值的点进行修正
for idx in shifted_indices:
    if idx == 0:
        restored_result[idx] = non_accumulated_result[idx]
    else:
        restored_result[idx] = restored_result[idx - 1]

# 绘制散点图
ax.scatter(range(len(restored_result)), restored_result, label="Non Accumulated Result")

ax.set_title('Non Accumulated Result')
ax.set_xlabel('Index')
ax.set_ylabel('Time')
ax.legend()

# 显示图形
plt.show()
