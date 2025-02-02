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
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """ getter for [] """
        return self.__W1

    @property
    def b1(self):
        """ getter for [] """
        return self.__b1

    @property
    def A1(self):
        """ getter for [] """
        return self.__A1

    @property
    def W2(self):
        """ getter for [] """
        return self.__W2

    @property
    def b2(self):
        """ getter for [] """
        return self.__b2

    @property
    def A2(self):
        """ getter for [] """
        return self.__A2

    def sigmoid(self, z):
        """ sigmoid function """
        return 1 / (1 + np.exp(-z))

    def forward_prop(self, X):
        """ performs one pass of forward propagation """
        z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = self.sigmoid(z1)
        z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = self.sigmoid(z2)
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """ cost of NN """
        inner1 = np.multiply(np.log(1.0000001 - A), (1 - Y))
        inner2 = np.multiply(np.log(A), Y) + inner1
        summa = np.sum(inner2)
        cel = (-1 / A.shape[1]) * summa
        return cel

    def evaluate(self, X, Y):
        """ evaluate networks predictions """
        _, hidden = self.forward_prop(X)
        res = np.where(hidden >= 0.5, 1, 0)
        return res, self.cost(Y, hidden)
