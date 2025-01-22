#!/usr/bin/env python3
"""
Script to calculate a Normal distribution
of continuous variables and discretes
"""

class Normal():
    """
    Class representing the Normal distribution
    """

    pi = 3.1415926536
    e = 2.7182818285

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        data: list of data points to estimate the distribution
        mean: mean of the distribution
        stddev: standard deviation of the distribution
        """
        if data is None:
            # If no data is given, use the provided mean and stddev
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            # If data is provided, calculate the mean and stddev based on data
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.mean = sum(data) / len(data)  # Mean of the data
                # Standard deviation calculation (sample stddev)
                variance = sum((x - self.mean) ** 2 for x in data) / len(data)
                self.stddev = variance ** 0.5

    def z_score(self, x):
        """
        Calculate the Z-score for a given x
        x: the value for which to calculate the Z-score
        return: the Z-score
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculate the x value from a given z-score
        z: the Z-score for which to calculate the x value
        return: the corresponding x value
        """
        return self.stddev * z + self.mean

    def pdf(self, x):
        """
        Calculate the Probability Density Function (PDF) for a given x
        x: the x value for which to calculate the PDF
        return: the PDF value
        """
        # Calculate the PDF using the normal distribution formula
        coefficient = 1 / (self.stddev * (2 * Normal.pi) ** 0.5)  # Normalizing factor
        exponent = -((x - self.mean) ** 2) / (2 * (self.stddev ** 2))  # Exponent part
        return coefficient * Normal.e ** exponent

    def cdf(self, x):
        """
        Calculate the Cumulative Distribution Function (CDF) for a given x
        x: the x value for which to calculate the CDF
        return: the CDF value
        """
        xa = (x - self.mean) / ((2 ** 0.5) * self.stddev)
        # Using the approximation for the error function (erf)
        errof = (((4 / Normal.pi) ** 0.5) * (xa - (xa ** 3) / 3 + (xa ** 5) / 10 - (xa ** 7) / 42 + (xa ** 9) / 216))
        cdf = (1 + errof) / 2
        return cdf
