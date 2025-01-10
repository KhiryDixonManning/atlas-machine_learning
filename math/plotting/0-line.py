#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3

#plot the graph
plt.xlim(0, 10)
plt.ylim(0, 11 ** 3)
plt.plot(y, color='red')
plt.show()