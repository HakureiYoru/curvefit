import numpy as np
from scipy.ndimage import distance_transform_edt


def xy_fft(gen_x, gen_y):
    # Fourier transform of gen_x
    ft_gen_x = np.fft.rfft(gen_x)
    abs_ft_gen_x = np.abs(ft_gen_x) / (len(gen_x) / 2)
    angle_ft_gen_x = np.angle(ft_gen_x)
    freq_gen_x = np.fft.rfftfreq(len(gen_x), d=1/20)

    # Fourier transform of gen_y
    ft_gen_y = np.fft.rfft(gen_y)
    abs_ft_gen_y = np.abs(ft_gen_y) / (len(gen_y) / 2)
    angle_ft_gen_y = np.angle(ft_gen_y)
    freq_gen_y = np.fft.rfftfreq(len(gen_y), d=1/20)

    results = {
        "gen_x_amplitudes": [],
        "gen_x_phases": [],
        "gen_y_amplitudes": [],
        "gen_y_phases": [],
        "gen_x_frequencies": [],
        "gen_y_frequencies": []
    }

    # Find the dominant frequency of gen_x
    for i in range(len(freq_gen_x)):
        if abs_ft_gen_x[i] > 0.2:  # set the boundary of important frequency components
            amplitude = abs_ft_gen_x[i]
            phase = angle_ft_gen_x[i]
            frequency = freq_gen_x[i]
            results["gen_x_amplitudes"].append(amplitude)
            results["gen_x_phases"].append(phase)
            results["gen_x_frequencies"].append(frequency)

    # Find the dominant frequency of gen_y
    for i in range(len(freq_gen_y)):
        if abs_ft_gen_y[i] > 0.2:  # set the boundary of important frequency components
            amplitude = abs_ft_gen_y[i]
            phase = angle_ft_gen_y[i]
            frequency = freq_gen_y[i]
            results["gen_y_amplitudes"].append(amplitude)
            results["gen_y_phases"].append(phase)
            results["gen_y_frequencies"].append(frequency)

    return results




def keep_one_period(gen_x, gen_y):
    # Calculate the Fourier Transform
    fourier_transform = np.fft.rfft(gen_y)
    abs_fourier_transform = np.abs(fourier_transform)
    power_spectrum = np.square(abs_fourier_transform)
    frequency = np.fft.rfftfreq(gen_y.size)
    dominant_frequency = frequency[np.argmax(power_spectrum)]
    period = int(np.round(1 / dominant_frequency))

    # Only keep one period of data
    return gen_x[:period], gen_y[:period]


def calculate_map(fit_x, fit_y, pixel_size, sigma, delta_time, progress_callback):
    larger_map = 0.2  # see definition below
    max_times_per_pixel = 100
    threshold = 0.1  # For  the weight map
    # Determine the range of the original coordinates
    x_min, x_max = np.min(fit_x) - larger_map, np.max(fit_x) + larger_map
    y_min, y_max = np.min(fit_y) - larger_map, np.max(fit_y) + larger_map

    detector_width = int(np.ceil((x_max - x_min) / pixel_size))
    detector_height = int(np.ceil((y_max - y_min) / pixel_size))

    progress_increment = 50.0 / 1000  # The range of the progress bar for this function is from 50% to 100%

    # Initialize the counts array
    counts = np.zeros((detector_height, detector_width))  # Note the reversed order

    time_map = np.full((detector_height, detector_width, max_times_per_pixel), np.nan)
    weight_time_map = np.full((detector_height, detector_width, max_times_per_pixel), np.nan)  # Add this line
    time_counter = np.zeros((detector_height, detector_width), dtype=int)

    # Create a binary image representing the trajectory
    trajectory_image = np.zeros((detector_height, detector_width))
    for x, y in zip(fit_x, fit_y):
        x_pixel = int(np.floor((x - x_min) / pixel_size))
        y_pixel = int(np.floor((y - y_min) / pixel_size))
        trajectory_image[y_pixel, x_pixel] = 1

    # Compute the distance transform
    dist = distance_transform_edt(1 - trajectory_image)

    # Compute the weight map from the distance transform
    weight_map = np.exp(-dist / sigma)

    # Count the number of fitted points at each pixel
    for i, (x, y) in enumerate(zip(fit_x, fit_y)):
        x_pixel = int(np.floor((x - x_min) / pixel_size))
        y_pixel = int(np.floor((y - y_min) / pixel_size))
        counts[y_pixel, x_pixel] += 1

        # Check if the weight is above a threshold
        if weight_map[y_pixel, x_pixel] > threshold * np.max(weight_map):
            # Copy time information to additional pixels
            for dx in range(-delta_time, delta_time + 1):
                for dy in range(-delta_time, delta_time + 1):
                    new_x_pixel = x_pixel + dx
                    new_y_pixel = y_pixel + dy

                    # Check if the additional pixel is within the detector boundaries
                    if 0 <= new_x_pixel < detector_width and 0 <= new_y_pixel < detector_height:
                        if time_counter[new_y_pixel, new_x_pixel] < max_times_per_pixel:
                            time_map[new_y_pixel, new_x_pixel, time_counter[new_y_pixel, new_x_pixel]] = i
                              # Use the weight of the new pixel
                            time_counter[new_y_pixel, new_x_pixel] += 1

        if progress_callback is not None:
            progress_callback(increment=progress_increment)

    # After all other calculations
    for y_pixel in range(detector_height):
        for x_pixel in range(detector_width):
            # Check if there are any non-nan values
            if not np.isnan(time_map[y_pixel, x_pixel]).all():
                # Calculate the real time of the particle in this pixel
                real_time = np.nanmean(time_map[y_pixel, x_pixel])

                # Calculate the weights for each timestamp
                time_diff = time_map[y_pixel, x_pixel] - real_time
                weight_time_map[y_pixel, x_pixel] = np.exp(-time_diff ** 2 / (2 * sigma ** 2))

    return counts, time_map, weight_map, weight_time_map, detector_width, detector_height, time_counter, x_min, y_min
