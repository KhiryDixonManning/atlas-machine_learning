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
        self.__L = len(layers)
        layers.insert(0, nx)
        self.__cache = {}
        self.__weights = {}
        self.__weights = {}
        for l in range(1, self.__L + 1):
            if layers[l] < 1:
                raise TypeError("layers must be a list of positive integers")
            he = np.random.randn(layers[l], layers[l - 1])
            self.weights["W" + str(l)] = he * np.sqrt(2.0 / (layers[l - 1]))
            self.__weights["b" + str(l)] = np.zeros((layers[l], 1))

    @property
    def L(self):
        """ getter for value of number of layers in network """
        return self.__L

    @property
    def cache(self):
        """ getter for cache containing all activation function outputs, as
            well as X """
        return self.__cache

    @property
    def weights(self):
        """ getter for  the dictionary containing entire network """
        return self.__weights

    def sigmoid(self, z):
        """ sigmoid function """
        return 1 / (1 + np.exp(-z))

    def forward_prop(self, X):
        """ one forward pass of neuron """
        self.__cache["A0"] = X
        A = X
        for l in range(1, self.__L + 1):
            W = self.__weights["W" + str(l)]
            b = self.__weights["b" + str(l)]
            z = np.dot(W, A) + b[0]
            A = self.sigmoid(z)
            self.__cache["A" + str(l)] = A
        return A, self.__cache

    def cost(self, Y, A):
        """ calculate the total cost of models output """
        inner1 = np.multiply(np.log(1.0000001 - A), (1 - Y))
        inner2 = np.multiply(np.log(A), Y) + inner1
        summa = np.sum(inner2)
        cel = (-(1 / Y.shape[1]) / A.shape[0]) * summa
        return cel

    def evaluate(self, X, Y):
        """ evaluate networks predictions """
        A, hidden = self.forward_prop(X)
        res = np.where(A >= 0.5, 1, 0)
        return res, self.cost(Y, A)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ one pass of backprogagation/gradient descent """
        N = Y.shape[1]
        leng = len(cache)
        W_cur = self.__weights["W" + str(leng - 1)]
        A_cur = cache["A" + str(leng - 1)]
        A_prev = cache["A" + str(leng - 2)]
        b_cur = self.__weights["b" + str(leng - 1)]

        adj = {}

        X = cache["A0"]
        dz2 = (A_cur - Y)
        dW2 = (1 / N) * np.dot(dz2, A_prev.T)
        db2 = (1 / N) * np.sum(dz2, keepdims=True)

        adj["W" + str(leng - 1)] = W_cur - alpha * dW2
        adj["b" + str(leng - 1)] = b_cur - alpha * db2

        for l in range(leng - 2, 0, -1):
            if l > 0:
                W_cur = self.__weights["W" + str(l)]
                W_prev = self.__weights["W" + str(l + 1)]
                A_cur = cache["A" + str(l)]
                A_prev = cache["A" + str(l - 1)]
                b_cur = self.__weights["b" + str(l)]
                dg = A_cur * (1 - A_cur)
                dz1 = (np.dot(W_prev.T, dz2)) * dg
                dw1 = (1 / N) * np.dot(dz1, A_prev.T)
                db1 = (1 / N) * np.sum(dz1, axis=1, keepdims=True)
                dz2 = dz1

                adj["W" + str(l)] = W_cur - alpha * dw1
                adj["b" + str(l)] = b_cur - alpha * db1

        self.__weights = adj
