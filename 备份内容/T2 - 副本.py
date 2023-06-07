import matplotlib.pyplot as plt
import numpy as np
import json


def gen(T, A, w, p1, p2, n):
    x = np.random.uniform(-T, T, n)
    t = np.arccos(x / T) / (w * np.pi) - p1 / (w * np.pi)
    y = A * np.cos(w * np.pi * t + p2) + np.random.normal(0, 0.1, n)
    return x, y


def run_gen(T, A, w, p1, p2, n):
    # Generate data
    x, y = gen(T, A, w, p1, p2, n)

    # Plot the result
    plt.figure(figsize=(10, 10))
    plt.scatter(x, y, label='Generated Data')
    plt.title('Generated Data')
    plt.show()

    # Convert data to list
    data1 = list(zip(x, y))

    # Store as dictionary
    data = {
        "Generated Data": data1,
    }

    # Write data to json file
    with open('output.json', 'w') as f:
        json.dump(data, f)

    # Write data to text file
    np.savetxt('output.dat', data1, fmt='%f')

    # Return the generated data
    return x, y
