#!/usr/bin/env python3
""" module cantaining function that evaluates the output of a neural network"""
import tensorflow.compat.v1 as tf


def evaluate(X, Y, save_path):
    """ function that evaluates the predicition of restored model

        Parameters:
            X [numpy.ndarry] -  input data to evaluate
            Y [numpy.ndarry] - one hot encoded output labels
            save_path [sting] - file path to load model from

        Returns:
            y_pred [numpy.ndarry] - the prediction of the network
            accuracy [float] - tensor containing the decimal accuracy
                                of the prediction
            loss [float] - tensor containing the loss of the prediction
        """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(f'{save_path}.meta')
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        y_pred = tf.get_collection("y_pred")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        saver.restore(sess, save_path)
        feed_dict = {x: X, y: Y}
        y_p, acc, los = sess.run([y_pred, accuracy, loss], feed_dict=feed_dict)
    return y_p, acc, los
