#!/usr/bin/env python3
""" module containing class that creates the class BayesianOptimization that
    performs Bayesian optimization on a noiseless 1D Gaussian process. """
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization():
    """ class that performs Bayesian optimization on a noiseless 1D Gaussian
        process.

            Attributes
            ----------
            f : function
                The black-box function.
            gp : GaussianProcess
                An instance of the class GaussianProcess.
            X_s : numpy.ndarray
                Acquisition sample points, evenly spaced between min and max,
                shape (ac_samples, 1).
            xsi : float
                Exploration-exploitation factor.
            minimize : bool
                Determines whether optimization should be performed for
                minimization or maximization.

    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """ creates the class BayesianOptimization that performs Bayesian
            optimization on a noiseless 1D Gaussian process.

                Parameters
                ----------
                f : function
                    The black-box function to be optimized.
                X_init : numpy.ndarray
                    Inputs already sampled with the black-box function,
                    shape (t, 1).
                Y_init : numpy.ndarray
                    Outputs of the black-box function for each input in X_init
                    shape (t, 1).
                bounds : tuple
                    Tuple of (min, max) representing the bounds of the space
                    in which to look for the optimal point.
                ac_samples : int
                    Number of samples that should be analyzed during
                    acquisition.
                l : float, optional
                    Length parameter for the kernel, default value is 1.
                sigma_f : float, optional
                    Standard deviation given to the output of the black-box
                    function, default value is 1.
                xsi : float, optional
                    Exploration-exploitation factor for acquisition, default
                    value is 0.01.
                minimize : bool, optional
                    Determines whether optimization should be performed for
                    minimization (True) or maximization (False), default value
                    is True.

                Attributes
                ----------
                f : function
                    The black-box function.
                gp : GaussianProcess
                    An instance of the class GaussianProcess.
                X_s : numpy.ndarray
                    Acquisition sample points, evenly spaced between min and
                    max, shape (ac_samples, 1).
                xsi : float
                    Exploration-exploitation factor.
                minimize : bool
                    Determines whether optimization should be performed for
                    minimization or maximization.
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.array([np.linspace(bounds[0], bounds[1], ac_samples)]).T
        self.xsi = xsi
        self.minimize = minimize
