import numpy as np
import json
import logging
from scipy.odr import Model, Data, ODR
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

# Configuring the logger
logger = logging.getLogger('filter_logger')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('filter_log.txt', 'w')  # Using 'w' mode to overwrite the log file
logger.addHandler(file_handler)


# Function to define parametric equations with cosine functions
def parametric_equations(t, params):
    # Unpacking parameters
    A1, A2, B1, B2, w1, w2, p1, p2, p3, p4 = params

    # Handling case where A2 is close to zero
    if np.abs(A2) < 1e-6:
        w2 = 0

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
def run_fit(x=None, y=None, params=None, filter_press_count=None, progress_callback=None):
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

    # Function to calculate parameter bounds for optimization
    def calculate_bounds(param_value, factor=0.5):
        # Set bounds to 0 if the parameter value is 0
        if param_value == 0:
            return (0, 0)
        # Calculate bounds based on the parameter value
        range_delta = factor * abs(param_value)
        lower_bound = param_value - 10
        upper_bound = param_value + 10
        # Swap if necessary
        if lower_bound > upper_bound:
            lower_bound, upper_bound = upper_bound, lower_bound
        return lower_bound, upper_bound

    # Create bounds for each parameter
    bounds = [calculate_bounds(param_value) for param_value in param_list]

    # Print debugging information
    # print("params:", params)
    # print("param_list:", param_list)
    # print("bounds:", bounds)

    # Perform the optimization using differential evolution
    result = differential_evolution(objective_function, bounds, args=(x, y), maxiter=1000, popsize=20,
                                    callback=progress_callback)

    # Extract the fitted parameters
    fitted_params = result.x

    # Generate the fitted x and y values
    t_fit = np.linspace(0, 10, 2000)
    x_fit, y_fit = parametric_equations(t_fit, fitted_params)

    # Estimate the time of generation for the observed x and y values
    t_obs_est = estimate_time(x, y, fitted_params)

    # Logging
    logger.info("------------")
    logger.info(f'Running run_fit function {filter_press_count} times.')
    logger.info(f"Input parameters: {params}")
    logger.info(f"Output parameters: {fitted_params}")
    logger.info("------------")
    logger.info(f"Fitted x: {x_fit}")
    logger.info(f"Fitted y: {y_fit}")
    logger.info(f"Estimated times: {t_obs_est}")

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
        "time_fit": t_obs_est.tolist()
    }


# Function to estimate the time of generation for each pair of observed x and y values
def estimate_time(x_obs, y_obs, fitted_params, t_range=(0, 10), resolution=1e-3):
    """
    Estimate the time of generation for each pair of observed x and y values based on the fitted parameters.

    Args:
        x_obs (np.array): The observed x values.
        y_obs (np.array): The observed y values.
        fitted_params (list): The parameters obtained from fitting.
        t_range (tuple): The range of time to search.
        resolution (float): The step size for the time search.

    Returns:
        np.array: The estimated times of generation.
    """
    estimated_times = []

    for x, y in zip(x_obs, y_obs):
        min_distance = float('inf')
        estimated_time = None

        # Loop through time and find the time that minimizes the distance to the observed points
        for t in np.arange(t_range[0], t_range[1], resolution):
            x_fit, y_fit = parametric_equations(t, fitted_params)
            distance = (x_fit - x) ** 2 + (y_fit - y) ** 2

            if distance < min_distance:
                min_distance = distance
                estimated_time = t

        estimated_times.append(estimated_time)

    # Normalize the estimated times
    estimated_times = np.array(estimated_times)
    estimated_times = (estimated_times - np.min(estimated_times)) / (np.max(estimated_times) - np.min(estimated_times))

    # Filter outliers
    return filter_outliers(estimated_times)


# Function to filter out outliers in data using a standard deviation criteria
def filter_outliers(data, m=3):
    mean = np.mean(data)
    std_dev = np.std(data)
    # Filtering data points that are within m standard deviations from the mean
    filtered_data = [x for x in data if (mean - m * std_dev < x < mean + m * std_dev)]
    return np.array(filtered_data)  # Convert the result to a NumPy array
