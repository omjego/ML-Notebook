from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
m = 500
n = 2
x = np.random.uniform(-4, 4, m)
y = x + np.random.standard_normal( m ) + 2.5
# plt.plot(x, y, 'o')


def cost(x, y, theta):
    res = 0.0
    for i in range(m):
        res += (theta[0] + theta[1]*x[i] - y[i])**2
    res /= (2.0 * m)
    return res


alpha = 0.3   # learning rate
theta = np.array([0, 0])  # initial values of theta
iterations = 1000  # number of iterations
cost_history = []
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


error= np.array([0 for _ in range(iterations)])
t1_history = np.array( [0 for _ in range(iterations)])
t2_history = np.array([0 for _ in range(iterations)])

for i in range(iterations):

    error[i] = cost(x, y, theta)
    t1_history[i] = theta[0]
    t2_history[i] = theta[1]

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
    theta = np.array([a, b])
    cost_history.append(cost(x, y, theta))
    #
    # if i == (iterations - 1):
    #     plt.plot(x, a*x + b)
    #     plt.title('Linear regression by gradient descent')
    #     plt.xlabel('x')
    #     plt.ylabel('y')

t0, t1 = np.meshgrid(t1_history, t2_history)
# Z = error.reshape(t0.shape)

ax.plot_surface(t0, t1, error)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')


print cost_history
print theta
plt.show()