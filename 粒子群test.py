import numpy as np
from geomdl import fitting
from geomdl.visualization import VisMPL as vis
import numpy as np
np.float = np.cfloat




def gen(T, A, B, w1, w2, p1, p2, n):
    # 随机生成n个范围在[0, T)的t值
    t = np.random.uniform(0, T, n)

    # 生成带有高斯的x和y值
    x = A * np.cos(w1 * np.pi * t + p1) + np.random.normal(0, 0.05, n)
    y = B * np.cos(w2 * np.pi * t + p2) + np.random.normal(0, 0.05, n)

    return x, y


# 生成数据
T = 1
A = 1
B = 1
w1 = 1
w2 = 1
p1 = 0
p2 = 1.5707963267948966
n = 50

x_gen, y_gen = gen(T, A, B, w1, w2, p1, p2, n)

# 数据拟合
points = np.array(list(zip(x_gen, y_gen)))
degree = 3  # 三次曲线拟合

# 全局曲线插值
curve_interpolation = fitting.interpolate_curve(points, degree)

# 全局曲线逼近
curve_approximation = fitting.approximate_curve(points, degree)

# 可视化插值结果
curve_interpolation.delta = 0.01
curve_interpolation.vis = vis.VisCurve2D()
curve_interpolation.render()

# 可视化逼近结果
curve_approximation.delta = 0.01
curve_approximation.vis = vis.VisCurve2D()
curve_approximation.render()
