import numpy as np
import json
import logging
from scipy.odr import Model, Data, ODR
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

# Configure logging
logger = logging.getLogger('filter_logger')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('filter_log.txt', 'w')  # Add 'w' mode for overwriting
logger.addHandler(file_handler)


def parametric_equations(t, params):
    A1, A2, B1, B2, w1, w2, p1, p2, p3, p4 = params

    if np.abs(A2) < 1e-6:
        w2 = 0

    x = A1 * np.cos(w1 * t + p1) + A2 * np.cos(w2 * t + p3)
    y = B1 * np.cos(w1 * t + p2) + B2 * np.cos(w2 * t + p4)
    return x, y


def objective_function(params, x_obs, y_obs):
    t_est = np.linspace(0, 10, len(x_obs) * 10)
    x_est, y_est = parametric_equations(t_est, params)
    x_est = np.interp(np.linspace(0, 10, len(x_obs)), t_est, x_est)
    y_est = np.interp(np.linspace(0, 10, len(x_obs)), t_est, y_est)
    return np.sum((x_est - x_obs) ** 2 + (y_est - y_obs) ** 2)


def run_fit(x=None, y=None, params=None, filter_press_count=None, progress_callback=None):
    A1 = params.get('A_x1', 0)
    A2 = params.get('A_x2', 0)
    B1 = params.get('B_y1', 0)
    B2 = params.get('B_y2', 0)
    w1 = params.get('f1', 0) * 2 * np.pi
    w2 = params.get('f2', 0) * 2 * np.pi
    p1 = params.get('p_x1', 0)
    p2 = params.get('p_y1', 0)
    p3 = params.get('p_x2', 0)
    p4 = params.get('p_y2', 0)
    param_list = [A1, A2, B1, B2, w1, w2, p1, p2, p3, p4]

    # 构建bounds
    def calculate_bounds(param_value, factor=0.5):
        # set bounds to 0 if param value is 0 (sometime we don't have multiple sin function
        if param_value == 0:
            return (0, 0)

        # calculate bounds by parameters
        range_delta = factor * abs(param_value)
        lower_bound = param_value - range_delta
        upper_bound = param_value + range_delta

        # Make sure lower_bound < upper_bound
        if lower_bound > upper_bound:
            lower_bound, upper_bound = upper_bound, lower_bound

        return lower_bound, upper_bound

    bounds = []
    for param_value in param_list:
        bounds.append(calculate_bounds(param_value))

    print("params:", params)
    print("param_list:", param_list)
    print("bounds:", bounds)

    result = differential_evolution(objective_function, bounds, args=(x, y), maxiter=1000, popsize=20,
                                    callback=progress_callback)

    fitted_params = result.x

    t_fit = np.linspace(0, 10, 2000)
    x_fit, y_fit = parametric_equations(t_fit, fitted_params)

    # Logging
    logger.info("------------")
    logger.info(f'Running run_fit function {filter_press_count} times.')
    logger.info(f"Input parameters: {params}")
    logger.info(f"Output parameters: {fitted_params}")
    logger.info("------------")
    logger.info(f"Fitted x: {x_fit}")
    logger.info(f"Fitted y: {y_fit}")

    # Save x_fit and y_fit to a JSON file
    output_data = {
        "x_fit": x_fit.tolist(),
        "y_fit": y_fit.tolist()
    }
    with open("output_new_xy.json", "w") as output_file:
        json.dump(output_data, output_file)

    return {
        "fitted_params": fitted_params.tolist(),
        "x_fit": x_fit.tolist(),
        "y_fit": y_fit.tolist()
    }
