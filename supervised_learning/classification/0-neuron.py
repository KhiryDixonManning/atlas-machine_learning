#!/usr/bin/env python3
""" module containing class Neuron """
import numpy as np


class Neuron():
    """class representing a single Neuron that performs binary classification

        Instance Attributes:
            W (normal dist) - the weights vector
            b (int) - the bias
            A (np.array?) - activated output"""

    def __init__(self, nx):
        """ initialize Neuron

            Parameters:
                nx (int) - nmber of input features """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx <= 0:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
