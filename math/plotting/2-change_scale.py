#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
change_scale = __import__('2-change_scale').change_scale

def change_scale():

    x = np.arange(0, 28651, 5730)
    r = np.log(0.5)
    t = 5730
    y = np.exp((r / t) * x)

    plt.plot(x, y)
    plt.xlabel('Time (years)')
    plt.ylabel('Fraction Remaining')
    plt.title("Exponential Decay of C-14")
    plt.yscale('log')
    plt.xlim(0, 28650)
    plt.show()
