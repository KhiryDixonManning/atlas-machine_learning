#!/usr/bin/env python3
""" module containing class NeuralNetwork """
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os.path


class DeepNeuralNetwork():
    """class representing a Deep Neural Network with multiple hidden layer that
        performs binary classification

        Instance Attributes:
            L (): number of layers
            cache (): all intermediary values
            weights (dict): all weights and biases
            """

    def __init__(self, nx,  layers, activation='sig'):
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
        if activation != 'sig' and activation != 'tanh':
            raise ValueError("activation must be 'sig' or 'tanh'")
        self.__L = len(layers)
        self.__activation = activation
        layers.insert(0, nx)
        self.__cache = {}
        self.__weights = {}
        self.__weights = {}
        for lay in range(1, self.__L + 1):
            if layers[lay] < 1:
                raise TypeError("layers must be a list of positive integers")
            he = np.random.randn(layers[lay], layers[lay - 1])
            layer = (layers[lay - 1])
            self.weights["W" + str(lay)] = he * np.sqrt(2.0 / layer)
            self.__weights["b" + str(lay)] = np.zeros((layers[lay], 1))

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

    @property
    def activation(self):
        """ getter for  activation function name """
        return self.__activation

    def sigmoid(self, z):
        """ sigmoid function """
        return 1 / (1 + np.exp(-z))

    def sig_deriv(self, A):
        """ derivative of sigmoid func"""
        return A * (1 - A)

    def tanh(self, x):
        """ tanh func"""
        return np.tanh(x)

    def tanh_derv(self, X):
        """ derivative of tanh func """
        return 1 - (np.tanh(X))**2

    def softmax(self, z):
        """ softmax function """
        ez = np.exp(z - np.max(z))
        return ez / np.sum(ez, axis=0, keepdims=True)

    def one_hot_encode(self, Y, classes):
        """ function that decodes a one hot encoded vector """
        if not isinstance(Y, np.ndarray) or not isinstance(classes, int):
            return None
        if classes < 2 or classes < Y[max(Y - 1)]:
            return None
        encoded = np.zeros((classes, len(Y)))
        for x, y in enumerate(Y):
            encoded[y][x] = 1
        return encoded

    def forward_prop(self, X):
        """ one forward pass of neuron """
        self.__cache["A0"] = X
        A = X
        for lay in range(1, self.__L + 1):
            W = self.__weights["W" + str(lay)]
            b = self.__weights["b" + str(lay)]
            z = np.dot(W, A) + b[0]
            if lay == self.__L:
                A = self.softmax(z)
            else:
                if self.__activation == 'sig':
                    A = self.sigmoid(z)
                else:
                    A = self.tanh(z)
            self.__cache["A" + str(lay)] = A
        return A, self.__cache

    def cost(self, Y, A):
        """ calculate the total cost of models output """
        m = Y.shape[1]
        cel = -np.sum(Y * np.log(A)) / m
        return cel

    def evaluate(self, X, Y):
        """ evaluate networks predictions """
        A, hidden = self.forward_prop(X)
        pred = np.argmax(A, axis=0)
        res = self.one_hot_encode(pred, Y.shape[0])
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
        db2 = (1 / N) * np.sum(dz2, axis=1, keepdims=True)

        adj["W" + str(leng - 1)] = W_cur - alpha * dW2
        adj["b" + str(leng - 1)] = b_cur - alpha * db2

        for lay in range(leng - 2, 0, -1):
            if lay > 0:
                W_cur = self.__weights["W" + str(lay)]
                W_prev = self.__weights["W" + str(lay + 1)]
                A_cur = cache["A" + str(lay)]
                A_prev = cache["A" + str(lay - 1)]
                b_cur = self.__weights["b" + str(lay)]
                if self.__activation == 'sig':
                    dg = self.sig_deriv(A_cur)
                else:
                    dg = self.tanh_derv(A_cur)
                dz1 = (np.dot(W_prev.T, dz2)) * dg
                dw1 = (1 / N) * np.dot(dz1, A_prev.T)
                db1 = (1 / N) * np.sum(dz1, axis=1, keepdims=True)
                dz2 = dz1

                adj["W" + str(lay)] = W_cur - alpha * dw1
                adj["b" + str(lay)] = b_cur - alpha * db1

        self.__weights = adj

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """ trains model fully """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step >= iterations:
                raise ValueError("step must be positive and <= iterations")
        steps = []
        costs = []

        for j in range(iterations):
            Y_hat, self.__cache = self.forward_prop(X)
            if verbose:
                if j == 0 or j % step == 0:
                    cost = self.cost(Y, Y_hat)
                    steps.append(j)
                    costs.append(cost)
                    print('cost after {} iterations: {}'.format(j, cost))
            self.gradient_descent(Y, self.__cache, alpha)

        A, cost = self.evaluate(X, Y)
        if graph:
            plt.plot(steps, costs)
            plt.title('Training Cost')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.show()
        return A, cost

    def save(self, filename):
        """ saves instance object in pickle file """
        if '.pkl' not in filename:
            filename = '{}.pkl'.format(filename)
        file = open(filename, 'wb')
        pickle.dump(self, file)
        file.close()

    @staticmethod
    def load(filename):
        """ loads a pickled DNN object"""
        if not os.path.isfile(filename):
            return None
        with open(filename, 'rb') as file:
            dnn = pickle.load(file)
        return dnn
