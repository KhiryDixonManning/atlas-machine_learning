#!/usr/bin/env python3
"""Module containing a function that fully trains a model and calculates the
   loss and accuracy."""

import tensorflow.compat.v1 as tf

calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes,
          activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """Function that fully trains a model.

    Parameters:
        X_train (numpy.ndarray): Training input data.
        Y_train (numpy.ndarray): Training labels.
        X_valid (numpy.ndarray): Validation input data.
        Y_valid (numpy.ndarray): Validation labels.
        layer_sizes (list): List containing the number of nodes in each layer
                             of the network.
        activations (list): List containing the activation functions for each
                             layer of the network.
        alpha (float): Learning rate.
        iterations (int): The number of iterations to train the model.
        save_path (str): File path to save the model to.

    Returns:
        str: The path where the model was saved.
    """
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    accuracy = calculate_accuracy(y, y_pred)
    loss = calculate_loss(y, y_pred)
    train_op = create_train_op(loss, alpha)

    # Add tensors to the collection for saving/restoring
    tf.compat.v1.add_to_collection('x', x)
    tf.compat.v1.add_to_collection('y', y)
    tf.compat.v1.add_to_collection('y_pred', y_pred)
    tf.compat.v1.add_to_collection('loss', loss)
    tf.compat.v1.add_to_collection('accuracy', accuracy)
    tf.compat.v1.add_to_collection('train_op', train_op)

    # Initialize variables and define the saver
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        diction = {x: X_train, y: Y_train}

        for i in range(iterations + 1):
            # Calculate the cost and accuracy for training and validation
            c_tloss = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            c_tacc = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
            c_vloss = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            c_vacc = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})

            # Print progress every 100 iterations or on the last iteration
            if i % 100 == 0 or i == iterations:
                print(f'After {i} iterations:')
                print(f'\tTraining Cost: {c_tloss}')
                print(f'\tTraining Accuracy: {c_tacc}')
                print(f'\tValidation Cost: {c_vloss}')
                print(f'\tValidation Accuracy: {c_vacc}')

            if i < iterations:
                # Run training operation
                trained = sess.run(train_op, feed_dict=diction)

        # Save the trained model
        save_p = saver.save(sess, save_path)

    return save_p
