#!/usr/bin/env python3
"""
Script to calculate a Poisson distribution
"""


class Poisson():
    """
    Class to represent a Poisson distribution and calculate its CDF and PMF.
    """

    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize method
        data: list of data points used to estimate the Poisson distribution
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
                # Calculate lambtha from data
                self.lambtha = sum(data) / len(data)

    def pmf(self, k):
        """
        Method to calculate the Probability Mass Function for Poisson
        k: integer value representing the number of occurrences (successes)
        return: PMF value for k
        """
        # Convert k to an integer if it is not already
        k = int(k)

        # If k is negative, return 0 since it's out of range for a Poisson distribution
        if k < 0:
            return 0

        # Calculate factorial of k
        factorial_k = 1
        for i in range(1, k + 1):
            factorial_k *= i

        # Calculate and return PMF
        pmf = Poisson.e ** -self.lambtha * self.lambtha ** k / factorial_k
        return pmf

    def cdf(self, k):
        """
        Method to calculate the Cumulative Distribution Function (CDF)
        k: integer value representing the number of occurrences (successes)
        return: CDF value for k
        """
        # Convert k to an integer if it is not already
        k = int(k)

        # If k is negative, return 0 since it's out of range for a Poisson distribution
        if k < 0:
            return 0

        # Calculate and return CDF by summing PMF values from 0 to k
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf
