import json
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def run_fit():
    with open('output.json', 'r') as f:
        data_dict = json.load(f)

    with open('parameters.json', 'r') as f:
        parameters = json.load(f)

    data = data_dict['Generated Data']
    t, x, y = zip(*data)
    t = np.array(t)
    x = np.array(x)
    y = np.array(y)


    n = parameters["n"]



    def gauss_function(t, A, w1, p1, B, w2, p2):
        x = A * np.cos(w1 * np.pi * t + p1)
        y = B * np.cos(w2 * np.pi * t + p2)
        return np.concatenate((x, y))

    initial_guess = [
        parameters["A"],
        parameters["w1"],
        parameters["p1"],
        parameters["B"],
        parameters["w2"],
        parameters["p2"]
    ]
    params, _ = curve_fit(gauss_function, t, np.concatenate((x, y)), p0=initial_guess)

    A_fit, w1_fit, p1_fit, B_fit, w2_fit, p2_fit = params

    fit_data = gauss_function(t, A_fit, w1_fit, p1_fit, B_fit, w2_fit, p2_fit)
    fit_x = fit_data[:n]
    fit_y = fit_data[n:]

    # 对拟合结果按 t 值进行排序
    sorted_indices = np.argsort(t)
    fit_x = fit_x[sorted_indices]
    fit_y = fit_y[sorted_indices]

    return fit_x, fit_y
