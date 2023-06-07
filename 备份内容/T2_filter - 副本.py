import json
import numpy as np
from scipy.odr import ODR, Model, Data
import matplotlib.pyplot as plt


def run_odr():
    with open('output.json', 'r') as f:
        data_dict = json.load(f)

    data = data_dict['Generated Data']
    x, y = zip(*data)
    x = np.array(x)
    y = np.array(y)

    # Here we use the normal distribution to generate the errors of x and y.
    # dx = np.random.normal(0, 0.01, size=n)
    # dy = np.random.normal(0, 0.01, size=n)
    dx = np.full_like(x, 0.01)
    dy = np.full_like(y, 0.01)

    # Define the model function.
    # This function takes the parameters beta
    # and x as input and returns the corresponding y value.
    # This is the format of the model function required for ODR.
    def f(beta, x):
        A, B, w, p1, p2 = beta
        # t = np.arccos(x / A) / (w * np.pi) - p1 / (w * np.pi)
        t = np.arccos(np.clip(x / A, -1, 1)) / (w * np.pi) - p1 / (w * np.pi)
        return B * np.cos(w * np.pi * t + p2)

    T2data = Data(x, y, dx, dy)
    T2model = Model(f)

    # Create an ODR object. We use the ODR class of the ODR package to create an ODR object
    # that connects the data to the model and can be used to run the ODR algorithm
    myodr = ODR(T2data, T2model, beta0=[1., 1., 1., 0., 0.])

    # Set the way the ODR object works.
    # We set fit_type to 0 to indicate that the ODR algorithm is used for fitting.
    myodr.set_job(fit_type=0)
    output = myodr.run()

    # If the algorithm does not converge,
    # we will run it again until convergence is reached or it has been run 100 times.
    if output.info != 1:
        print("\nRestart ODR until convergence is reached")
        i = 1
        while output.info != 1 and i < 100:
            print("restart", i)
            output = myodr.restart()
            i += 1

    # Output results. We print the reason for stopping the ODR algorithm,
    # the final parameter values, the message flags,
    # the standard deviation of the parameters
    # and the square root of the diagonal elements of the covariance matrix.
    print("ODR results:")
    print("------------")
    print("   stop reason:", output.stopreason)
    print("        params:", output.beta)
    print("          info:", output.info)
    print("       sd_beta:", output.sd_beta)
    print("sqrt(diag(cov):", np.sqrt(np.diag(output.cov_beta)))

    x_fit = np.linspace(min(x), max(x), 1000)
    y_fit = f(output.beta, x_fit)

    plt.figure(figsize=(10, 10))
    plt.scatter(x, y, label='Data')
    plt.plot(x_fit, y_fit, label='ODR fit')
    plt.title('ODR fit to data')
    plt.legend()
    plt.show()

    x_fit = np.linspace(min(x), max(x), 1000)
    y_fit = f(output.beta, x_fit)

    return x_fit, y_fit
