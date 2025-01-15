#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def frequency():

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    bins = range(0, 101, 10)

    plt.hist(student_grades, bins=bins, edgecolor='black')
    plt.xlabel('Grades')
    plt.ylabel("Number of Students")
    plt.title("Project A")
    plt.xlim(0, 100)
    plt.ylim(0, 30)
    plt.show()

    print("Plot generated successfully!")  # This is 32 characters long


# Call the function to generate the plot
frequency()
