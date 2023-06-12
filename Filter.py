import numpy as np
import json
from scipy.odr import Model, Data, ODR


def run_fit():
    # Load data and parameters from files
    try:
        with open('output.json', 'r') as f:
            data_dict = json.load(f)
        with open('parameters.json', 'r') as f:
            params = json.load(f)
    except FileNotFoundError as fnf_error:
        print(fnf_error)
        return
    except json.JSONDecodeError as json_error:
        print(json_error)
        return

    # Get x and y data from 'Generated Data'
    x, y = np.array(data_dict['Generated Data']).T

    beta_orig = np.array([params[param] for param in ["A", "B", "w1", "w2", "p1", "p2"]])
    beta_limit = (0.2, 0.2, 0.2, 0.2, 0.2, 0.2)  # 20%

    # Define the model function
    def f(beta, x):
        A, B, w1, w2, p1, p2 = beta
        dA, dB, dw1, dw2, dp1, dp2 = beta_limit
        # Ensure that the parameter values are within the predefined ranges.
        # If the parameters are outside the limits, the fit may be unreliable or not as expected
        # Check parameter ranges (upper bound/ lower bound)
        param_check = (A >= beta_orig[0] - dA) & (A <= beta_orig[0] + dA) & \
                      (B >= beta_orig[1] - dB) & (B <= beta_orig[1] + dB) & \
                      (w1 >= beta_orig[2] - dw1) & (w1 <= beta_orig[2] + dw1) & \
                      (w2 >= beta_orig[3] - dw2) & (w2 <= beta_orig[3] + dw2) & \
                      (p1 >= beta_orig[4] - dp1) & (p1 <= beta_orig[4] + dp1) & \
                      (p2 >= beta_orig[5] - dp2) & (p2 <= beta_orig[5] + dp2)

        if not np.all(param_check):
            return 0  # Return 0 if any parameter is out of range
        # if out of range, no continue calculation
        t0 = ((0 * np.pi + np.arccos(np.clip(x / A, -1, 1))) - p1) / w1
        t1 = ((1 * np.pi + np.arccos(np.clip(x / A, -1, 1))) - p1) / w1
        y0calc = B * np.cos(w2 * t0 + p2)
        y1calc = B * np.cos(w2 * t1 + p2)
        return np.where(np.abs(y - y0calc) < np.abs(y - y1calc), y0calc, y1calc)

    # Define the data and the model for ODR fitting
    T2data = Data(x, y, np.full_like(x, 0.01), np.full_like(y, 0.01))
    T2model = Model(f)

    # Run ODR
    myodr = ODR(T2data, T2model, beta0=beta_orig, ifixb=[1, 1, 1, 1, 1, 1])
    myodr.set_job(fit_type=0)
    output = myodr.run()

    # If not converged, rerun until convergence or reach the maximum count of 100
    i = 1
    while output.info != 1 and i < 100:
        print(f"Restarting ODR, iteration {i}")
        output = myodr.restart()
        i += 1

    # Print the result
    print("ODR results:")
    print("------------")
    print("Stop reason:", output.stopreason)
    print("Parameters:", output.beta)
    # This is the information code for the ODR fit. In general, info=1 indicates a successful fit.
    print("Info:", output.info)
    print("Standard Deviation of Beta:", output.sd_beta)
    print("Square root of Diagonal of Covariance:", np.sqrt(np.diag(output.cov_beta)))

    # Define fitted y
    fit_y = f(output.beta, x)

    # Return fitted parameters and fitted y values
    return {
        "A": output.beta[0],
        "B": output.beta[1],
        "w1": output.beta[2],
        "w2": output.beta[3],
        "p1": output.beta[4],
        "p2": output.beta[5],
        "fit_x": x,
        "fit_y": fit_y
    }
