#!/usr/bin/env python3
"""Module containing a function that evaluates the output of a neural network."""

import tensorflow.compat.v1 as tf


def evaluate(X, Y, save_path):
    """Function that evaluates the prediction of a restored model.

    Parameters:
        X (numpy.ndarray): Input data to evaluate.
        Y (numpy.ndarray): One-hot encoded output labels.
        save_path (str): File path to load the model from.

    Returns:
        tuple: A tuple containing:
            - y_pred (numpy.ndarray): The prediction of the network.
            - accuracy (float): The decimal accuracy of the prediction.
            - loss (float): The loss of the prediction.
    """
    with tf.Session() as sess:
        # Load the saved model's meta graph
        saver = tf.train.import_meta_graph(f'{save_path}.meta')

        # Retrieve tensors from the graph
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        y_pred = tf.get_collection("y_pred")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]

        # Restore the saved model
        saver.restore(sess, save_path)

        # Prepare the feed dictionary and run the session
        feed_dict = {x: X, y: Y}
        y_p, acc, los = sess.run([y_pred, accuracy, loss], feed_dict=feed_dict)

    return y_p, acc, los
