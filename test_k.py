import json
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load data from file
with open('test.json', 'r') as f:
    data = json.load(f)

data = data['Test Data']
x, y = zip(*data)
x = np.array(x).reshape(-1, 1)
y = np.array(y).reshape(-1, 1)

# Apply polynomial regression
degree = 3  # Set the degree of the polynomial
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(x)
regressor = LinearRegression()
regressor.fit(X_poly, y)

# Generate points for plotting the fitted curve
x_fit = np.linspace(min(x), max(x), 100).reshape(-1, 1)
X_fit_poly = poly_features.transform(x_fit)
y_fit = regressor.predict(X_fit_poly)

# Visualize the result
plt.scatter(x, y, label='Data Points')
plt.plot(x_fit, y_fit, color='red', label='Fitted Curve')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
