import numpy as np
import json


def gen(T, A, B, w1, w2, p1, p2, n):

    t = np.random.uniform(0, T, n)
    x = A * np.cos(w1 * np.pi * t + p1) + np.random.normal(0, 0.01, n)
    y = B * np.cos(w2 * np.pi * t + p2) + np.random.normal(0, 0.01, n)

    return x, y


def run_gen(T, A, B, w1, w2, p1, p2, n):

    x, y = gen(T, A, B, w1, w2, p1, p2, n)

    data = {
        "Generated Data": list(zip(x, y)),
    }

    with open('output.json', 'w') as f:
        json.dump(data, f)

    parameters = {"T": T, "A": A, "B": B, "w1": w1, "w2": w2, "p1": p1, "p2": p2, "n": n}
    with open('parameters.json', 'w') as f:
        json.dump(parameters, f)

    return x, y
