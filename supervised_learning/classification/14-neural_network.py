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
        """ getter for W1 """
        return self.__W1

    @property
    def b1(self):
        """ getter for b1 """
        return self.__b1

    @property
    def A1(self):
        """ getter for A1 """
        return self.__A1

    @property
    def W2(self):
        """ getter for W2 """
        return self.__W2

    @property
    def b2(self):
        """ getter for b2 """
        return self.__b2

    @property
    def A2(self):
        """ getter for A2 """
        return self.__A2

    def sigmoid(self, z):
        """ sigmoid activation function """
        return 1 / (1 + np.exp(-z))

    def forward_prop(self, X):
        """ performs one forward pass"""
        z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = self.sigmoid(z1)
        z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = self.sigmoid(z2)
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """ calculates the total loss of the network """
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

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """ one pass oof backproagation/gradient discent  """
        N = X.shape[1]

        dldw2 = (1 / N) * np.dot((A2 - Y), A1.T)
        dldb2 = (1 / N) * np.sum((A2 - Y), keepdims=True)
        dldg = A1 * (1 - A1)
        dldz1 = ((self.__W2.T * (A2 - Y)) * dldg)
        dldw1 = (1 / N) * np.dot(dldz1, X.T)
        dldb1 = (1 / N) * np.sum(dldz1, keepdims=True, axis=1)

        self.__W1 -= alpha * dldw1
        self.__b1 -= alpha * dldb1
        self.__W2 -= alpha * dldw2
        self.__b2 -= alpha * dldb2

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """ trains model fully """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        for j in range(iterations):
            Y_hat = self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)

        A, cost = self.evaluate(X, Y)
        return A, cost
