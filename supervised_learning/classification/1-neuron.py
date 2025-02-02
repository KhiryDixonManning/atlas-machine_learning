#!/usr/bin/env python3
""" module containing class Neuron """
import numpy as np


class Neuron():
    """class representing a single Neuron that performs binary classification

        Instance Attributes:
            __W (normal dist) - the weights vector
            __b (int) - the bias
            __A (np.array?) - activated output"""

    def __init__(self, nx):
        """ initialize Neuron

            Parameters:
                nx (int) - nmber of input features """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx <= 0:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ getter for weight """
        return self.__W

    @property
    def b(self):
        """ getter for bias """
        return self.__b

    @property
    def A(self):
        """ getter of activation output """
        return self.__A
