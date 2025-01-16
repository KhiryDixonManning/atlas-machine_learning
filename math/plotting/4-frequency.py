#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))
    bins = range(0, 101, 10)

    plt.hist(student_grades, bins=bins, edgecolor='k')
    plt.xlabel('Grades')
    plt.ylabel("Number of Students")
    plt.title("Project A")

    # Set x-axis ticks to count by 10s (from 0 to 100)
    plt.xticks(np.arange(0, 101, 10))

    # Set y-axis ticks to count by 5s (from 0 to 30)
    plt.yticks(np.arange(0, 31, 5))

    # Set axis limits
    plt.xlim(0, 100)
    plt.ylim(0, 30)

    plt.show()
