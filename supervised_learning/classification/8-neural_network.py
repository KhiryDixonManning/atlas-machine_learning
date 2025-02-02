#!/usr/bin/env python3
""" module containing class NeuralNetwork """
import numpy as np


class NeuralNetwork():
    """class representing a Neural Network with one hidden layer that
        performs binary classification

        Instance Attributes:
            W1 (normal dist) - the weights vector for hidden layer
            b1 (int) - the bias for the hidden layer
            A1 (np.array?) - activated output for the hidden layer
            W2 (normal dist) - the weights vector for output layer
            b2 (int) - the bias for the output layer
            A2 (np.array?) - activated output for the output layer"""

    def __init__(self, nx, nodes):
        """ initialize Neural Network

            Parameters:
                nx (int) - nmber of input features
                nodes (int) - number of nodes in the hidden layer """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx <= 0:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes <= 0:
            raise ValueError("nodes must be a positive integer")
        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0
