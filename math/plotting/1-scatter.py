#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def scatter():

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180
    plt.figure(figsize=(6.4, 4.8))  # Optional: setting the figure size (default is 6.4 x 4.8 inches)

    # Scatter plot
    plt.scatter(x, y, color='m')  # Plot the data as magenta points
    plt.xlabel('Height (in)')  # Set x-axis label
    plt.ylabel('Weight (lbs)')  # Set y-axis label
    plt.title("Men's Height vs Weight")  # Set the title

    plt.show()  # Display the plot

# Call the scatter function to display the plot
scatter()
