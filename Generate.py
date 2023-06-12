import matplotlib.pyplot as plt
import numpy as np
import json


def gen(T, A, B, w1, w2, p1, p2, n):
    # 随机生成n个范围在[0, T)的t值
    t = np.random.uniform(0, T, n)

    # 生成带有高斯的x和y值
    x = A * np.cos(w1 * np.pi * t + p1) + np.random.normal(0, 0.02, n)  # mean,variance,count
    y = B * np.cos(w2 * np.pi * t + p2) + np.random.normal(0, 0.02, n)

    return x, y


def run_gen(T, A, B, w1, w2, p1, p2, n):
    # Generate data
    x, y = gen(T, A, B, w1, w2, p1, p2, n)

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


    # Save parameters to json file
    parameters = {"T": T, "A": A, "B": B, "w1": w1, "w2": w2, "p1": p1, "p2": p2, "n": n}
    with open('parameters.json', 'w') as f:
        json.dump(parameters, f)

    print("x length:", len(x))
    print("y length:", len(y))

    # Return the generated data
    return x, y
