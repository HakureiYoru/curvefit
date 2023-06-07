import json
import numpy as np
from scipy.odr import ODR, Model, Data
import matplotlib.pyplot as plt

# Use JSON given by Generate.py
with open('output.json', 'r') as f:
    data_dict = json.load(f)

# Converted to a numpy array for later computation.
data = data_dict['Generated Data']
x, y = zip(*data)
x = np.array(x)
y = np.array(y)
n = len(x)
print ("array x = len ",n)
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
    A, B, w1, w2, p1, p2 = beta
    # The effective input of the arc cosine can only be between -1, +1
    t0 = ((0*np.pi + np.arccos(np.clip(x / A, -1, 1))) - p1)/w1   #calc possible extra t vals by adding n*pi in to the acos result
    t1 = ((1*np.pi + np.arccos(np.clip(x / A, -1, 1))) - p1)/w1
    
    y0calc = B * np.cos(w2 * t0 + p2)                             #possible ycalc = f(t) values
    y1calc = B * np.cos(w2 * t1 + p2)
    y0diff = (y-y0calc)*(y-y0calc)                                #calc squares of differences with measured y 
    y1diff = (y-y1calc)*(y-y1calc)
    np.putmask(y0calc,y1diff<y0diff,y1calc)                       #if y1calc is closer to y than y0calc, replace 
    return y0calc

#This is the function that generates parametrically from t.
def gen(beta,n,T):
    A, B, w1, w2, p1, p2 = beta
    t = np.linspace(0,T,num=n)
    xgen = A * np.cos(w1 * t + p1)
    ygen = B * np.cos(w2 * t + p2)
    return xgen, ygen



T2data = Data(x, y, dx, dy)
T2model = Model(f)

# Create an ODR object. We use the ODR class of the ODR package to create an ODR object
# that connects the data to the model and can be used to run the ODR algorithm
#myodr = ODR(T2data, T2model, beta0=[2, 2, 2*np.pi/50, 2*np.pi/50, 0, np.pi / 2])
myodr = ODR(T2data, T2model, beta0=[ 2, 2, 2*np.pi/64, 2*np.pi/64, -np.pi/2, 0],ifixb=[1,1,1,1,1,0])

myodr.set_job(fit_type=0)
output = myodr.run()

# If the algorithm does not converge,
# we will run it again until convergence is reached, or it has been run 100 times.
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


#calculate the line based on the parameters
x_fit,y_fit = gen(output.beta,1000,32)

plt.figure(figsize=(10, 10))
plt.xlim([-2.5, 2.5])
plt.ylim([-2.5, 2.5])
plt.scatter(x, y, label='Data')
plt.plot(x_fit, y_fit, label='ODR fit')
plt.title('ODR fit to data')
plt.legend()
plt.show()
