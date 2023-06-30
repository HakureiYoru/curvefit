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



def process_data(gen_x, gen_y, f1, f2):

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

    # Adjusting the frequency according to the number of points in a cycle
    base_points = 500
    adjustment_factor = base_points / len(gen_x)

    # Adjust the frequency because we only use one period of data
    f1 = f1 / adjustment_factor
    f2 = f2 / adjustment_factor

    return {"Phase difference": phase_difference_in_pi, "f1": f1, "f2": f2}


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
