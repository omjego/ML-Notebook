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
x = np.c_[np.ones(m), x]
theta = np.linalg.pinv(x.T.dot(x)).dot(x.T).dot(y)   # ((xT * x) ^ - 1) . xT . Y

plt.plot(x[:, 1], theta.T.dot(x.T).T, 'r')
plt.title('Gradient descent')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
