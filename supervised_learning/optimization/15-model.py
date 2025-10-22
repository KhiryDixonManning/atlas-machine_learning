#!/usr/bin/env python3
"""Module containing functions to build, train,
   and save a neural network using Adam optimization, mini-batch
   gradient descent, learning rate decay, and batch normalization."""

import tensorflow.compat.v1 as tf
import numpy as np


def forward_prop(prev, layers, activations, epsilon):
    """Performs forward propagation pass through the network.

    Parameters:
        prev (tensor): The activated output of the previous layer.
        layers (list): List of integers representing the number of units for each layer.
        activations (list): List of activation functions for each layer.
        epsilon (float): Small number for numerical stability in batch normalization.

    Returns:
        tensor: The final output after forward propagation.
    """
    for lay in range(len(layers)):
        if lay != len(layers) - 1:
            prev = create_layer(prev, layers[lay], activations[lay], epsilon)
        else:
            prev = create_last_layer(prev, layers[lay])
    return prev


def create_layer(prev, n, activation, epsilon):
    """Creates a hidden layer with batch normalization and activation.

    Parameters:
        prev (tensor): The output from the previous layer.
        n (int): Number of units in this layer.
        activation (function): The activation function to apply after batch normalization.
        epsilon (float): Small number for numerical stability in batch normalization.

    Returns:
        tensor: The activated output of this layer.
    """
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(units=n, kernel_initializer=init)
    x = layer(prev)

    mean, variance = tf.nn.moments(x, axes=[0])
    gamma = tf.Variable(tf.ones([n]))
    beta = tf.Variable(tf.zeros([n]))
    norm = tf.nn.batch_normalization(x, mean, variance, beta, gamma, epsilon)

    return tf.keras.layers.Activation(activation)(norm)


def create_last_layer(prev, n):
    """Creates the last layer (without batch normalization) for the neural network.

    Parameters:
        prev (tensor): The output from the previous layer.
        n (int): Number of units in this layer.

    Returns:
        tensor: The output of this layer.
    """
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(units=n, kernel_initializer=init)
    return layer(prev)


def shuffle_data(X, Y):
    """Shuffles the dataset (X, Y) in the same way for both inputs and labels.

    Parameters:
        X (np.ndarray): Features with shape (m, nx).
        Y (np.ndarray): Labels with shape (m, ny).

    Returns:
        tuple: Shuffled features and labels.
    """
    perm = np.random.permutation(X.shape[0])
    return X[perm], Y[perm]


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """Builds, trains, and saves a neural network using Adam optimization,
       mini-batch gradient descent, learning rate decay, and batch normalization.

    Parameters:
        Data_train (tuple): Training data (X_train, Y_train).
        Data_valid (tuple): Validation data (X_valid, Y_valid).
        layers (list): List of integers representing the number of units for each layer.
        activations (list): List of activation functions for each layer.
        alpha (float): Learning rate (default: 0.001).
        beta1 (float): Adam optimizer beta1 (default: 0.9).
        beta2 (float): Adam optimizer beta2 (default: 0.999).
        epsilon (float): Small value to prevent division by zero in Adam optimizer (default: 1e-8).
        decay_rate (float): Rate of learning rate decay (default: 1).
        batch_size (int): Batch size for mini-batch gradient descent (default: 32).
        epochs (int): Number of epochs to train the model (default: 5).
        save_path (str): Path to save the trained model (default: '/tmp/model.ckpt').

    Returns:
        str: The path to the saved model.
    """
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    # Initialize placeholders for input and output
    x = tf.placeholder(dtype=tf.float32, shape=[None, X_train.shape[1]], name="x")
    y = tf.placeholder(dtype=tf.float32, shape=[None, Y_train.shape[1]], name="y")

    # Add placeholders to collection
    tf.compat.v1.add_to_collection('x', x)
    tf.compat.v1.add_to_collection('y', y)

    # Forward propagation
    y_pred = forward_prop(x, layers, activations, epsilon)
    tf.compat.v1.add_to_collection('y_pred', y_pred)

    # Compute loss and accuracy
    loss = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)
    tf.compat.v1.add_to_collection('loss', loss)

    pred = tf.math.argmax(y_pred, axis=1)
    act = tf.math.argmax(y, axis=1)
    equality = tf.math.equal(pred, act)
    accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))
    tf.compat.v1.add_to_collection('accuracy', accuracy)

    # Initialize global step for learning rate decay
    global_step = tf.Variable(0, trainable=False)
    steps = round(X_train.shape[0] / batch_size)
    decay_step = steps * epochs

    # Learning rate decay operation
    alpha = tf.train.inverse_time_decay(alpha, global_step, decay_step, decay_rate, staircase=True)

    # Adam optimizer and training operation
    optim = tf.train.AdamOptimizer(learning_rate=alpha, beta1=beta1, beta2=beta2, epsilon=epsilon)
    train_op = optim.minimize(loss, global_step)
    tf.compat.v1.add_to_collection('train_op', train_op)

    # Initialize variables and saver
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        t_feed_dict = {x: X_train, y: Y_train}
        v_feed_dict = {x: X_valid, y: Y_valid}

        # Training loop
        for i in range(epochs):
            e_t_acc = sess.run(accuracy, feed_dict=t_feed_dict)
            e_t_loss = sess.run(loss, feed_dict=t_feed_dict)
            e_v_acc = sess.run(accuracy, feed_dict=v_feed_dict)
            e_v_loss = sess.run(loss, feed_dict=v_feed_dict)

            print(f"After {i} epochs:")
            print(f"\tTraining Cost: {e_t_loss}")
            print(f"\tTraining Accuracy: {e_t_acc}")
            print(f"\tValidation Cost: {e_v_loss}")
            print(f"\tValidation Accuracy: {e_v_acc}")

            # Shuffle data and perform mini-batch gradient descent
            X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)
            batches = 0
            num_samples = X_train.shape[0]
            last_batch = num_samples % batch_size
            if last_batch == 0:
                last_batch = batch_size
            steps = (num_samples - last_batch) // batch_size

            for j in range(0, int(steps) + 1):
                if j != steps:
                    X_batch = X_shuffled[batches:batches + batch_size]
                    Y_batch = Y_shuffled[batches:batches + batch_size]
                    batches += batch_size
                else:
                    X_batch = X_shuffled[batches:batches + last_batch]
                    Y_batch = Y_shuffled[batches:batches + last_batch]

                feed_dict = {x: X_batch, y: Y_batch}
                sess.run(train_op, feed_dict=feed_dict)
                acc = sess.run(accuracy, feed_dict=feed_dict)
                los = sess.run(loss, feed_dict=feed_dict)

                if j % 100 == 0:
                    print(f"\tStep {j}:")
                    print(f"\t\tTraining Cost: {los}")
                    print(f"\t\tTraining Accuracy: {acc}")

        # Final evaluations after training
        e_t_acc = sess.run(accuracy, feed_dict=t_feed_dict)
        e_t_loss = sess.run(loss, feed_dict=t_feed_dict)
        e_v_acc = sess.run(accuracy, feed_dict=v_feed_dict)
        e_v_loss = sess.run(loss, feed_dict=v_feed_dict)

        print(f"After {epochs} epochs:")
        print(f"\tTraining Cost: {e_t_loss}")
        print(f"\tTraining Accuracy: {e_t_acc}")
        print(f"\tValidation Cost: {e_v_loss}")
        print(f"\tValidation Accuracy: {e_v_acc}")

        # Save the model
        save_path = saver.save(sess, save_path)

    return save_path
