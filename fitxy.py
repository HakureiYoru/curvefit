import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def sum_of_sines(t, A1, w1, p1, B1, w2, p2):
    return A1 * np.sin(w1 * t + p1) + B1 * np.sin(w2 * t + p2)

def fit_gen_x_and_gen_y(t_values, gen_x, gen_y):
    initial_guess_x = [1, 1, 0, 1, 1, 0]
    initial_guess_y = [1, 1, 0, 1, 1, 0]

    try:
        params_x, _ = curve_fit(sum_of_sines, t_values, gen_x, p0=initial_guess_x, maxfev=10000)
        params_y, _ = curve_fit(sum_of_sines, t_values, gen_y, p0=initial_guess_y, maxfev=10000)
    except Exception as e:
        print("Error in curve fitting:", e)
        return None

    A1_x, w1_x, p1_x, B1_x, w2_x, p2_x = params_x
    A1_y, w1_y, p1_y, B1_y, w2_y, p2_y = params_y

    # Plotting the fitted curves and original data
    # For gen_x
    plt.figure(figsize=(10, 4))
    plt.plot(t_values, gen_x, 'b-', label='Original gen_x')
    plt.plot(t_values, sum_of_sines(t_values, A1_x, w1_x, p1_x, B1_x, w2_x, p2_x), 'r--', label='Fitted gen_x')
    plt.xlabel('Time')
    plt.ylabel('gen_x')
    plt.legend()
    plt.title('Fitting for gen_x')
    plt.show()

    # For gen_y
    plt.figure(figsize=(10, 4))
    plt.plot(t_values, gen_y, 'b-', label='Original gen_y')
    plt.plot(t_values, sum_of_sines(t_values, A1_y, w1_y, p1_y, B1_y, w2_y, p2_y), 'r--', label='Fitted gen_y')
    plt.xlabel('Time')
    plt.ylabel('gen_y')
    plt.legend()
    plt.title('Fitting for gen_y')
    plt.show()

    return {
        "gen_x_params": {"A1_x": A1_x, "w1_x": w1_x, "p1_x": p1_x, "B1_x": B1_x, "w2_x": w2_x, "p2_x": p2_x},
        "gen_y_params": {"A1_y": A1_y, "w1_y": w1_y, "p1_y": p1_y, "B1_y": B1_y, "w2_y": w2_y, "p2_y": p2_y}
    }
