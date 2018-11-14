import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#cost function
def cost_function(X, y, theta):
    y_predict = np.dot(X, theta)
    cost = 0.5 * np.mean(np.square(y_predict-y))
    return cost

# 梯度下降的算法
def gradient_descent(X, y, theta, alpha, iterations):

    cost_line = np.zeros(iterations)
    for i in range(iterations):

        last_theta = theta
        for index, each_theta in enumerate(last_theta):
            h = np.dot(X, last_theta)
            each_theta = each_theta - alpha * np.mean((h - y) * X[:, [index]])
            theta[index] = each_theta

        # m = y.size
        # h = np.dot(X, theta)
        # theta = theta - alpha * (1/m) * np.dot(X.T, h-y)
        cost_line[i] = cost_function(X, y, theta)
        print('cost =', cost_line[i])
    return cost_line


# using pandas read file
data = pd.read_csv('ex1data1.txt',header = None)
# print(data)

# using numpy read file
# data_numpy = np.fromfile('ex1data1.txt',dtype = np.float)
# print(data_numpy)

# setting x, y
x = np.loadtxt('ex1data1.txt',dtype = np.float,delimiter=",",usecols=(0,))
y = np.loadtxt('ex1data1.txt',dtype = np.float,delimiter=",",usecols=(1,))
print(y[0])
y = y.reshape(y.size,1)
print(y[0])

# Add a column of ones to X and initialize fitting parameters
X = np.hstack((np.ones((x.size,1)), x.reshape(x.size, 1)))
theta = np.zeros((2, 1))
iterations = 1500
alpha = 0.01

cost_line = gradient_descent(X, y, theta, alpha, iterations)


# plot the cost line
plt.figure()
plt.plot(cost_line)
plt.ylabel('Cost')
plt.xlabel('Iterations')
plt.show()


# # plotting the points
# plt.figure()
# plt.scatter(x,y)
# plt.show()


