#!/usr/bin/env python3
""" module containing class that represents a noiseless 1D gaussian
    process """
import numpy as np


class GaussianProcess():
    """ represents a noiseless 1d gaussian process

        Attributes
        ----------
        X : numpy.ndarray
            Inputs, shape (t, 1).
        Y : numpy.ndarray
            Outputs, shape (t, 1).
        l : float
            Length parameter for the kernel.
        sigma_f : float
            Standard deviation given to the output of the black-box
            function.
        K : numpy.ndarray
            Current covariance kernel matrix for the Gaussian process.

        Methods
        -------
        kernel : input mat1, input mat2
            function that calculates the covariance kernel matrix between
            two matrices using the Radial Basis Function (RBF) kernel
        predict : numpy.ndarray
            predicts the mean and standard deviation of points in a
            Gaussian process.
        update: numpy.ndarray, numpy.ndarray
            updates the Gaussian Process with new sample point and function
            value.
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """ Create the class GaussianProcess that represents a noiseless 1D
            Gaussian process. This class initializes a noiseless 1D Gaussian
            process

            Parameters
            ----------
            X_init : numpy.ndarray
                Inputs already sampled with the black-box function,
                shape (t, 1).
            Y_init : numpy.ndarray
                Outputs of the black-box function for each input in X_init,
                shape (t, 1).
            l : float, optional
                Length parameter for the kernel, default value is 1.
            sigma_f : float, optional
                Standard deviation given to the output of the black-box
                function, default value is 1.

            Attributes
            ----------
            X : numpy.ndarray
                Inputs, shape (t, 1).
            Y : numpy.ndarray
                Outputs, shape (t, 1).
            l : float
                Length parameter for the kernel.
            sigma_f : float
                Standard deviation given to the output of the black-box
                function.
            K : numpy.ndarray
                Current covariance kernel matrix for the Gaussian process.
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """ Calculates the covariance kernel matrix between two matrices using
            the Radial Basis Function (RBF) kernel.

            Parameters
            ----------
            X1 : numpy.ndarray
                Input matrix 1, shape (m, 1).
            X2 : numpy.ndarray
                Input matrix 2, shape (n, 1).

            Returns
            -------
            K : numpy.ndarray
                Covariance kernel matrix, shape (m, n).
        """

        dist_mat = (np.array([np.sum(X1 ** 2, 1)]).T + np.sum(
            X2 ** 2, 1)) - (2 * np.dot(X1, X2.T))
        K = self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * dist_mat)
        return K

    def predict(self, X_s):
        """ predicts the mean and standard deviation of points in a
            Gaussian process.

            Parameters
            ----------
            X_s : numpy.ndarray
                Points whose mean and standard deviation should be
                calculated, shape (s, 1).

            Returns
            -------
            mu : numpy.ndarray
                Mean for each point in X_s, shape (s,).
            sigma : numpy.ndarray
                Variance for each point in X_s, shape (s,).
        """
        k = self.kernel(self.X, self.X) + np.square(1e-8) * np.eye(len(self.X))
        k_cv = self.kernel(self.X, X_s)
        k_prior = self.kernel(X_s, X_s)
        k_inv = np.linalg.inv(k)
        mu = np.dot(np.dot(k_cv.T, k_inv), self.Y)
        cov = k_prior - np.dot(np.dot(k_cv.T, k_inv), k_cv)
        return mu.T[0], np.diag(cov)

    def update(self, X_new, Y_new):
        """ updates the Gaussian Process with new sample point and function
            value.

            Parameters
            ----------
            X_new : numpy.ndarray
                New sample point, shape (1,).
            Y_new : numpy.ndarray
                New sample function value, shape (1,).
        """
        self.X = np.concatenate((self.X, np.array([X_new])), axis=0)
        self.Y = np.concatenate((self.Y, np.array([Y_new])), axis=0)

        self.K = self.kernel(self.X, self.X)
