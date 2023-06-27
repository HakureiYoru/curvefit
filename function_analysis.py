import numpy as np


def analyze_function(gen_x, gen_y):
    # Fourier transform of gen_x
    ft_gen_x = np.fft.rfft(gen_x)
    abs_ft_gen_x = np.abs(ft_gen_x) / (len(gen_x) / 2)
    angle_ft_gen_x = np.angle(ft_gen_x)
    freq_gen_x = np.fft.rfftfreq(len(gen_x))

    # Fourier transform of gen_y
    ft_gen_y = np.fft.rfft(gen_y)
    abs_ft_gen_y = np.abs(ft_gen_y) / (len(gen_y) / 2)
    angle_ft_gen_y = np.angle(ft_gen_y)
    freq_gen_y = np.fft.rfftfreq(len(gen_y))

    analysis_result = {
        "gen_x": [],
        "gen_y": []
    }

    # Find the dominant frequency of gen_x
    for i in range(len(freq_gen_x)):
        if abs_ft_gen_x[i] > 0.2:  # set the boundary of important frequency components
            analysis_result["gen_x"].append({
                "Frequency": len(gen_x) / (1 / freq_gen_x[i]),
                "Amplitude": abs_ft_gen_x[i],
                "Phase": angle_ft_gen_x[i]
            })

    # Find the dominant frequency of gen_y
    for i in range(len(freq_gen_y)):
        if abs_ft_gen_y[i] > 0.2:  # set the boundary of important frequency components
            analysis_result["gen_y"].append({
                "Frequency": len(gen_y) / (1 / freq_gen_y[i]),
                "Amplitude": abs_ft_gen_y[i],
                "Phase": angle_ft_gen_y[i]
            })

    return analysis_result


def process_data(gen_x, gen_y):
    # Find A and B
    max_x = np.max(gen_x)
    min_x = np.min(gen_x)
    A = (max_x - min_x) / 2

    max_y = np.max(gen_y)
    min_y = np.min(gen_y)
    B = (max_y - min_y) / 2

    # Find phase difference
    peak_index_x = np.argmax(gen_x)
    peak_index_y = np.argmax(gen_y)

    index_difference = peak_index_y - peak_index_x
    phase_difference = (2 * np.pi * index_difference / len(gen_x))

    # If the phase difference is negative, convert it to a positive value
    if phase_difference < 0:
        phase_difference += 2 * np.pi

    if phase_difference > np.pi:
        phase_difference = 2 * np.pi - phase_difference

    # Output phase difference in π format
    phase_difference_in_pi = round(phase_difference / np.pi, 2)

    return {"A": A, "B": B, "Phase difference": phase_difference_in_pi}


def keep_one_period(gen_x, gen_y, t_measured):
    # Calculate the Fourier Transform
    fourier_transform = np.fft.rfft(gen_y)
    abs_fourier_transform = np.abs(fourier_transform)
    power_spectrum = np.square(abs_fourier_transform)
    frequency = np.fft.rfftfreq(gen_y.size)

    # Find the frequency with the maximum power
    dominant_frequency = frequency[np.argmax(power_spectrum)]

    # Compute the period as the inverse of the frequency
    period = int(np.round(1 / dominant_frequency))

    # calculate number of periods and points in a period
    num_periods = int(np.round(len(gen_y) * dominant_frequency))

    points_per_period = int(np.round(len(gen_y) / num_periods))

    # print the total point read, number of periods, number of points per period
    print("-------------------")
    print("Total point read: ", len(gen_y))
    print("Number of periods: ", num_periods)
    print("Number of points per period: ", points_per_period)

    # Only keep one period of data
    return gen_x[:period], gen_y[:period], t_measured[:period]
