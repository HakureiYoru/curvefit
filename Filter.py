import numpy as np
import json
import logging
from scipy.optimize import differential_evolution
from scipy.ndimage.filters import median_filter

# Configuring the logger
logger = logging.getLogger('filter_logger')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('filter_log.txt', 'w')  # Using 'w' mode to overwrite the log file
logger.addHandler(file_handler)


# Function to define parametric equations with cosine functions
def parametric_equations(t, params):
    # Unpacking parameters
    A1, A2, B1, B2, w1, w2, p1, p2, p3, p4 = params

    # Defining parametric equations
    x = A1 * np.cos(w1 * t + p1) + A2 * np.cos(w2 * t + p3)
    y = B1 * np.cos(w1 * t + p2) + B2 * np.cos(w2 * t + p4)
    return x, y


# Objective function for optimization which computes the sum of squared differences
def objective_function(params, x_obs, y_obs):
    # Create a time array for estimation
    t_est = np.linspace(0, 10, len(x_obs) * 10)
    # Compute the estimated x and y using the parametric equations
    x_est, y_est = parametric_equations(t_est, params)
    # Interpolate the estimated x and y to the same length as x_obs
    x_est = np.interp(np.linspace(0, 10, len(x_obs)), t_est, x_est)
    y_est = np.interp(np.linspace(0, 10, len(x_obs)), t_est, y_est)
    # Compute the sum of squared differences
    return np.sum((x_est - x_obs) ** 2 + (y_est - y_obs) ** 2)


# Main function to perform the fitting
def run_fit(f2_real = None, f1_real = None, x=None, y=None, params=None, bounds_factor_dict=None, filter_press_count=None, progress_callback=None):
    # Extracting parameters from the input dictionary
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
    param_names = ['A_x1', 'A_x2', 'B_y1', 'B_y2', 'f1', 'f2', 'p_x1', 'p_y1', 'p_x2', 'p_y2']  # Order matters

    # Function to calculate parameter bounds for optimization
    def calculate_bounds(param_name, param_value):
        factor = bounds_factor_dict.get(param_name, 0.5)  # Default factor is 0.5 if not in the dict
        # Set bounds to 0 if the parameter value is 0
        if param_value == 0:
            return 0, 0
        # Calculate bounds based on the parameter value
        range_delta = factor * abs(param_value)
        lower_bound = param_value - range_delta
        upper_bound = param_value + range_delta
        # Swap if necessary
        if lower_bound > upper_bound:
            lower_bound, upper_bound = upper_bound, lower_bound
        return lower_bound, upper_bound

    # Create bounds for each parameter
    bounds = [calculate_bounds(param_name, param_value) for param_name, param_value in zip(param_names, param_list)]

    # Perform the optimization using differential evolution
    result = differential_evolution(objective_function, bounds, args=(x, y), maxiter=1000, popsize=20,
                                    callback=progress_callback)

    # Extract the fitted parameters
    fitted_params = result.x

    # Assume f2_real and f1_real are defined somewhere
    min_freq = min(f2_real, f1_real)

    # Check if the minimal frequency is not zero to avoid division by zero
    if min_freq != 0:
        # Calculate the real number of time points
        real_time_points = (1 / min_freq) * 1000

        # Generate the fitted x and y values
        t_fit = np.linspace(0, 10, int(real_time_points))

    x_fit, y_fit = parametric_equations(t_fit, fitted_params)

    # Note that this time is really just for end0.
    # And the role of end was supposed to be just to derive the result of the fit parameter.
    # There is no need to calculate the time.

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
        "y_fit": y_fit.tolist(),
    }




