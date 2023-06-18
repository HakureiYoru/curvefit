import numpy as np
import json
import logging
from scipy.odr import Model, Data, ODR

# Configure logging
logger = logging.getLogger('filter_logger')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('filter_log.txt', 'w')  # Add 'w' mode for overwriting
logger.addHandler(file_handler)


def run_fit(x=None, y=None, params=None, beta_limit_dict=None, ifixb=None, filter_press_count=None):
    if x is None or y is None or params is None:
        # logger.info("No load, Find generate")
        try:
            with open('output.json', 'r') as f:
                data_dict = json.load(f)
            with open('parameters.json', 'r') as f:
                params = json.load(f)
        except FileNotFoundError as fnf_error:
            logger.error(fnf_error)
            return
        except json.JSONDecodeError as json_error:
            logger.error(json_error)
            return

        x, y = np.array(data_dict['Generated Data']).T
    else:
        print("Loaded data")

    beta_orig = np.array([params[param] for param in ["A", "B", "w1", "w2", "p1", "p2"]])
    beta_limit = [beta_limit_dict[param] for param in ["A", "B", "w1", "w2", "p1", "p2"]]

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

    # In this example, the uncertainty in the x and y axes is set to 0.01 for all data points.
    T2data = Data(x, y, np.full_like(x, 0.01), np.full_like(y, 0.01))
    T2model = Model(f)
    myodr = ODR(T2data, T2model, beta0=beta_orig, ifixb=ifixb)
    # ifixb This list determines which parameters in the model should be variable and which should be fixed during
    # the fitting process.
    myodr.set_job(fit_type=0)
    output = myodr.run()

    i = 1
    while output.info != 1 and i < 20:
        output = myodr.restart()
        i += 1

    chi_squared = output.sum_square
    logger.info("------------")
    logger.info(f'Running run_fit function {filter_press_count} times.')
    logger.info("ODR results:")
    logger.info(f"Input parameters: {params}")
    logger.info(f"Parameter limits: {beta_limit_dict}")
    logger.info(f"Output parameters: {output.beta}")
    logger.info(f"Stop reason: {output.stopreason}")
    logger.info(f"Chi: {chi_squared}")
    logger.info("------------")
    return {
        "A": output.beta[0],
        "B": output.beta[1],
        "w1": output.beta[2],
        "w2": output.beta[3],
        "p1": output.beta[4],
        "p2": output.beta[5],
        "chi": chi_squared
    }
