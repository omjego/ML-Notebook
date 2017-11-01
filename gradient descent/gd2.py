import numpy as np
import matplotlib.pyplot as plt

m = 100
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
x = np.c_[np.ones(m), x]
i = 0
while i < iterations:
    updated = np.zeros(n)
    for j in range(n):
        updated[j] = theta[j] - alpha * 1 / float(m) * np.sum((theta.T.dot(x.T).T - y) * x[:, j])
    theta = updated
    i += 1
plt.plot(x[:,1], theta.T.dot(x.T).T, 'g')


# print theta.T
# print x[:, 0]
# for i in range(iterations):
#     # update theta 0
#     partial = 0.0
#     for j in range(m):
#         partial += (theta[0] + theta[1]*x[j] - y[j])
#     partial /= m
#     a = theta[0] - alpha * partial
#
#     # update theta 1
#     partial = 0.0
#     for j in range(m):
#         partial += x[j] * (theta[0] + theta[1] * x[j] - y[j])
#     partial /= m
#     b = theta[1] - alpha * partial
#     theta = np.array([a, b])
#     if i == (iterations - 1):
#         plt.plot(x, a*x + b)

plt.title('Linear regression by gradient descent')
plt.xlabel('x')
plt.ylabel('y')
print theta
plt.show()


def gradient_descent(theta, x, y, alpha):
    iterations = 100
    i = 0
    while i < iterations :
        updated = np.zeros(n)
        for j in range(n):
            updated[j] = theta[j] - alpha * 1 / float(m) * np.sum((theta.T.dot(x.T).T - y) * x[:, j])
        theta = updated
        i += 1
        if i == (iterations - 1):
            plt.plot(x, theta.T.dot(x.T).T)

