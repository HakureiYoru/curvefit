import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 定义二维高斯函数模型
def gauss_function(t, A, w1, p1, B, w2, p2):
    x = A * np.cos(w1 * np.pi * t + p1) + np.random.normal(0, 0.01, n)
    y = B * np.cos(w2 * np.pi * t + p2) + np.random.normal(0, 0.01, n)
    return np.concatenate((x, y))

# 生成样本数据
n = 100
t = np.linspace(0, 1, n)  # 均匀的，实际上应该是random.uniform???
A = 1.0
w1 = 2.0
p1 = 0
B = 0.5
w2 = 4.0
p2 = np.pi/2
data = gauss_function(t, A, w1, p1, B, w2, p2)
x = data[:n]
y = data[n:]


initial_guess = [A, w1, p1, B, w2, p2]  # 初始猜测值,应该要从外部读取的，不然每次要手动设置
params, _ = curve_fit(gauss_function, t, np.concatenate((x, y)), p0=initial_guess)

# 输出拟合结果
A_fit, w1_fit, p1_fit, B_fit, w2_fit, p2_fit = params


plt.figure()
plt.plot(x, y, 'b.', label='Original Data')
fit_data = gauss_function(t, A_fit, w1_fit, p1_fit, B_fit, w2_fit, p2_fit)
fit_x = fit_data[:n]
fit_y = fit_data[n:]
plt.plot(fit_x, fit_y, 'r-', label='Fitted Curve')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
