import numpy as np
import json
import logging
from scipy.odr import Model, Data, ODR

# Configure logging
logger = logging.getLogger('filter_logger')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('filter_log.txt', 'w')  # Add 'w' mode for overwriting
logger.addHandler(file_handler)


def calculate_t_result(beta, x, y):
    A, B, w1, w2, p1, p2 = beta
    t_result = [0]
    for i in range(1, len(x)):
        x_prev, x_cur = x[i - 1], x[i]
        y_cur = y[i]

        t_prev = t_result[-1]
        t0 = ((0 * np.pi + np.arccos(np.clip(x_cur / A, -1, 1))) - p1) / w1 + t_prev
        t1 = ((1 * np.pi + np.arccos(np.clip(x_cur / A, -1, 1))) - p1) / w1 + t_prev

        y0calc = B * np.cos(w2 * t0 + p2)
        y1calc = B * np.cos(w2 * t1 + p2)

        t_cur = t0 if abs(y_cur - y0calc) < abs(y_cur - y1calc) else t1

        t_result.append(t_cur)

    return t_result


def run_fit(x=None, y=None, params=None, beta_limit_dict=None, ifixb=None, filter_press_count=None):
    required_keys = ["A", "B", "w1", "w2", "p1", "p2"]
    for key in required_keys:
        if key not in params:
            params[key] = 0

    for key in required_keys:
        if key not in beta_limit_dict:
            beta_limit_dict[key] = [0, 0]

    beta_orig = np.array([params[param] for param in required_keys])
    beta_limit = [beta_limit_dict[param] for param in required_keys]

    def f(beta, x):
        A, B, w1, w2, p1, p2 = beta
        dA, dB, dw1, dw2, dp1, dp2 = beta_limit
        param_check = (A >= beta_orig[0] - dA) & (A <= beta_orig[0] + dA) & \
                      (B >= beta_orig[1] - dB) & (B <= beta_orig[1] + dB) & \
                      (w1 >= beta_orig[2] - dw1) & (w1 <= beta_orig[2] + dw1) & \
                      (w2 >= beta_orig[3] - dw2) & (w2 <= beta_orig[3] + dw2) & \
                      (p1 >= beta_orig[4] - dp1) & (p1 <= beta_orig[4] + dp1) & \
                      (p2 >= beta_orig[5] - dp2) & (p2 <= beta_orig[5] + dp2)

        if not np.all(param_check):
            return 0

        t0 = ((0 * np.pi + np.arccos(np.clip(x / A, -1, 1))) - p1) / w1
        t1 = ((1 * np.pi + np.arccos(np.clip(x / A, -1, 1))) - p1) / w1
        y0calc = B * np.cos(w2 * t0 + p2)
        y1calc = B * np.cos(w2 * t1 + p2)
        return np.where(np.abs(y - y0calc) < np.abs(y - y1calc), y0calc, y1calc)

    T2data = Data(x, y, np.full_like(x, 0.01), np.full_like(y, 0.01))
    T2model = Model(f)
    myodr = ODR(T2data, T2model, beta0=beta_orig, ifixb=ifixb)
    myodr.set_job(fit_type=0)
    output = myodr.run()

    i = 1
    while output.info != 1 and i < 20:
        output = myodr.restart()
        i += 1

    chi_squared = output.sum_square

    E = f(output.beta, x)
    O = y
    chi_squared_per_point = (O - E) ** 2 / (E + 1e-6)

    t_result = calculate_t_result(output.beta, x, y)

    with open('t_result.json', 'w') as json_file:
        json.dump(t_result, json_file)

    logger.info("------------")
    logger.info(f'Running run_fit function {filter_press_count} times.')
    logger.info("ODR results:")
    logger.info(f"Input parameters: {params}")
    logger.info(f"Parameter limits: {beta_limit_dict}")
    logger.info(f"Output parameters: {output.beta}")
    logger.info(f"Stop reason: {output.stopreason}")
    logger.info(f"Chi: {chi_squared}")
    logger.info(f"Chi per point: {chi_squared_per_point}")
    logger.info("------------")



    return {
        "A": output.beta[0],
        "B": output.beta[1],
        "w1": output.beta[2],
        "w2": output.beta[3],
        "p1": output.beta[4],
        "p2": output.beta[5],
        "chi": chi_squared,
        "chi_per_point": chi_squared_per_point.tolist(),
        "fitted time": t_result
    }
