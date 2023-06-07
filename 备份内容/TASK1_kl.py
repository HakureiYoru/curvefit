import numpy as np
from scipy.odr import ODR, Model, Data, RealData
import matplotlib.pyplot as plt
import json


# 定义散点数据的生成函数
def gen(n,T, A, w, p1, p2):
    t = np.random.uniform(0, T/4, n)
    x = A * np.cos(w * t + p1)+np.random.normal(0,0.01,n)
    y = A * np.cos(w * t + p2)+np.random.normal(0,0.01,n)
    return x, y


# Example: quarter circle
#x, y = gen(32,128, 2, 2, 2*np.pi/128, 2*np.pi/128, -np.pi/2, 0)

# Example: semi circle
#x, y = gen(32,128, 2, 2, 2*np.pi/64, 2*np.pi/64, -np.pi/2, 0)

# Example: semi circle
x, y = gen(64,128, 1, 2*np.pi/32, -np.pi/4, 0)


# plot it
plt.figure(figsize=(10, 10))
plt.scatter(x, y, label='Original data')
plt.title('Generated Data')
plt.legend()
plt.show()

data1 = list(zip(x, y))
data = {
    "Generated Data": data1,
}
# In JSON
with open('output.json', 'w') as f:
    json.dump(data, f)

# In text
np.savetxt('output.dat', data1, fmt='%f')
