#!/usr/bin/env python3
import numpy as np  # imports NumPy library and allows you to refer to it
import matplotlib.pyplot as plt  # plotting library for Python


def scatter():
    np.random.seed(5)

mean = [69, 0]  # a list mean with two elements
cov = [[15, 8], [8, 15]]  # a 2x2 list cov
""" generates 2000 random samples from multivariate normal distribution."""
x, y = np.random.multivariate_normal(mean, cov, 2000).T
y += 180  # adds 180 to every element in the y array

plt.scatter(x, y, color='m')  # creates a scatter plot using matplotlib
plt.xlabel('height(in)')  # xaxis is height
plt.ylabel('weight(lbs)')  # yaxis is weight
plt.title("Men's Height vs Weight")  # Title of scatter plot
plt.show()
