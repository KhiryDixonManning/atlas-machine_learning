#!/usr/bin/env python3
""" module containing function that calculates accuracy of predictions """
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def calculate_accuracy(y, y_pred):
    """ function that calculates accuracy of a TensorFlow model's predicitions

        Parameters:
            y [symbtensor] - placeholder for the labels of the input data
            y_pred [tensor] - tensor containing the networks predictions

        Returns:
            [tensor] - tensor containing the decimal accuracy of the prediction
        """
    pred = tf.math.argmax(y_pred, axis=1)
    act = tf.math.argmax(y, axis=1)
    equality = tf.math.equal(pred, act)
    return tf.math.reduce_mean(tf.cast(equality, tf.float32))
