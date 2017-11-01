"""
 Created by Omkar Jadhav
"""

import numpy as np
import matplotlib.pyplot as plt

m = 100  # number of training examples
n = 2  # number of parameters
x = np.random.uniform(-4, 4, m)
y = x + np.random.standard_normal(m) + 2.5
plt.plot(x, y, 'bo')
alpha = 0.1   # learning rate
theta = np.array([0, 0])  # initial values of theta
iterations = 1000  # number of iterations
x = np.c_[np.ones(m), x]
i = 0
while i < iterations:
    updated = np.zeros(n)
    for j in range(n):
        updated[j] = theta[j] - alpha * 1 / float(m) * np.sum((theta.T.dot(x.T).T - y) * x[:, j])
    theta = updated
    i += 1

plt.plot(x[:, 1], theta.T.dot(x.T).T, 'r')
plt.title('Gradient descent')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
