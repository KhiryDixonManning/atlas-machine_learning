#!/usr/bin/env python3
"""
Script to calculate an Exponential distribution
"""

class Exponential():
    """
    Class to represent an Exponential distribution and calculate its CDF and PDF.
    """

    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize method
        data: list of data points used to estimate the Exponential distribution
        lambtha: expected number of occurrences in a given time frame
        """
        if data is None:
            # Use given lambtha if data is None
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            # Check if data is provided
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                # Calculate lambtha as the reciprocal of the mean of the data
                self.lambtha = 1 / (sum(data) / len(data))

    def pdf(self, x):
        """
        Method to calculate the Probability Density Function (PDF)
        x: value at which to calculate the PDF
        return: PDF value
        """
        if x < 0:
            return 0
        else:
            # Calculate PDF: λ * e^(-λ * x)
            pdf = self.lambtha * (Exponential.e ** (-self.lambtha * x))
            return pdf

    def cdf(self, x):
        """
        Method to calculate the Cumulative Distribution Function (CDF)
        x: value at which to calculate the CDF
        return: CDF value
        """
        if x < 0:
            return 0
        # Calculate CDF: 1 - e^(-λ * x)
        return 1 - (Exponential.e ** (-self.lambtha * x))
