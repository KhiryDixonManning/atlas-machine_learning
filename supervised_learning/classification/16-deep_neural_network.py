#!/usr/bin/env python3
""" module containing class NeuralNetwork """
import numpy as np


class DeepNeuralNetwork():
    """class representing a Deep Neural Network with multiple hidden layer that
        performs binary classification

        Instance Attributes:
            L (): number of layers
            cache (): all intermediary values
            weights (dict): all weights and biases
            """

    def __init__(self, nx,  layers):
        """ initialize Deep Neural Network

            Parameters:
                nx (int) - nmber of input features
                layers (int) - number of nodes in the hidden layer """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        layers.insert(0, nx)
        self.cache = {}
        self.weights = {}
        self.weights = {}
        for l in range(1, self.L + 1):
            if layers[l] < 1:
                raise TypeError("layers must be a list of positive integers")
            he = np.random.randn(layers[l], layers[l - 1])
            self.weights["W" + str(l)] = he * np.sqrt(2.0 / (layers[l - 1]))
            self.weights["b" + str(l)] = np.zeros((layers[l], 1))
