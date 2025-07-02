#!/usr/bin/env python3
""" tbw """
import tensorflow as tf


def autoencoder(input_dims, filters, latent_dims):
    """ tbw """

    def sample(params):
        """ tbw """
        z_mean, z_logsig = params
        eps = tf.keras.backend.random_normal(
            shape=(
                tf.keras.backend.shape(z_mean)[0],
                tf.keras.backend.int_shape(z_mean)[1]),
            mean=0.0,
            stddev=0.1)
        out = z_mean + tf.keras.backend.exp(z_logsig)

        return out * eps

    layers = len(filters)
    decode_layers = filters[::-1]

    # encoder
    encode_inpt = tf.keras.layers.Input(shape=(input_dims,))
    layer = encode_inpt
    for i in range(layers):
        layer = tf.keras.layers.Dense(units=filters[i],
                                      activation='relu')(layer)
    z_mean = tf.keras.layers.Dense(units=latent_dims)(layer)
    z_logsig = tf.keras.layers.Dense(units=latent_dims)(layer)
    points = tf.keras.layers.Lambda(sample, output_shape=(
        latent_dims,))([z_mean, z_logsig])
    encode = tf.keras.Model(
        inputs=encode_inpt, outputs=[
            z_mean, z_logsig, points])

    # decoder
    inpt = tf.keras.layers.Input(shape=(latent_dims,))
    layer = inpt
    for i in range(layers):
        layer = tf.keras.layers.Dense(units=decode_layers[i],
                                      activation='relu')(layer)
    latent = tf.keras.layers.Dense(
        units=input_dims,
        activation='sigmoid')(layer)
    decode = tf.keras.Model(inputs=inpt, outputs=latent)

    # full autoencoder
    autoencode = tf.keras.Model(inputs=encode_inpt,
                                outputs=decode(encode(encode_inpt)[2]))

    r_loss = tf.keras.losses.binary_crossentropy(
        encode_inpt, latent) * input_dims
    kl_div = tf.keras.backend.sum(
        1 + z_logsig -
        tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_logsig),
        axis=-1) * -0.5
    total_loss = tf.keras.backend.mean(r_loss + kl_div)

    # # compile
    # autoencode.compile(optimizer='adam', loss=total_loss)

    return encode, decode, autoencode
