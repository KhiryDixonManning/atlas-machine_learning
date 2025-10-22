#!/usr/bin/env python3
""" module containing function that determines if you should stop gradient
    descent early """


def early_stopping(cost, opt_cost, threshold, patience, count):
    """ function that determines if you should stop gradient descent early


        PARAMETERS
        ==========
        cost [?]: current validation cost of the neural network

        opt_cost [?]: lowest recorded validation cost of the neural network

        threshold [?]: threshold used for early stopping

        patience [?]: patience count used for early stopping

        count [?]: count of how long the threshold has not been met


        RETURNS
        =======
        [boolean]: whether the network should be stopped early, followed
                    by the updated count
    """
    count = 0 if opt_cost - cost > threshold else (count + 1)
    boolean = True if count == patience else False
    return boolean, count
