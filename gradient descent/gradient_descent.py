import numpy as np
import matplotlib.pyplot as plt

m = 500
n = 2
x = np.random.uniform(-4, 4, m)
y = x + np.random.standard_normal( m ) + 2.5
plt.plot(x, y, 'o')


def cost(x, y, theta):
    res = 0.0
    for i in range(m):
        res += (theta[0] + theta[1]*x[i] - y[i])**2
    res /= (2.0 * m)
    return res


alpha = 0.1   # learning rate
theta = np.array([0, 0])  # initial values of theta
iterations = 1000  # number of iterations

for i in range(iterations):
    # update theta 0
    partial = 0.0
    for j in range(m):
        partial += (theta[0] + theta[1]*x[j] - y[j])
    partial /= m
    a = theta[0] - alpha * partial

    # update theta 1
    partial = 0.0
    for j in range(m):
        partial += x[j] * (theta[0] + theta[1] * x[j] - y[j])
    partial /= m
    b = theta[1] - alpha * partial
    print theta[0] - a
    theta = np.array([a, b])
    if i == (iterations - 1):
        plt.plot(x, a*x + b)

plt.title('Linear regression by gradient descent')
plt.xlabel('x')
plt.ylabel('y')
print theta
plt.show()