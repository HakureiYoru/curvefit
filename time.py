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
plt.figure()

# 绘制非累加结果
plt.scatter(range(len(non_accumulated_result)), non_accumulated_result, label="Non Accumulated Result")

# 添加图例
plt.legend()

# 添加标题和轴标签
plt.title('Results from t_result.json')
plt.xlabel('Index')
plt.ylabel('Time')

# 显示图形
plt.show()

# 计算累加结果的斜率
if accumulated_result:
    indexes = np.arange(len(accumulated_result))
    slope, intercept = np.polyfit(indexes, accumulated_result, 1)
    print(f"The slope of the accumulated result is: {slope}")
else:
    print("No accumulated_result data found.")
