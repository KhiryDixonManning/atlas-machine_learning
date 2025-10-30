#!/usr/bin/env python3
""" module containing function that contructs a vanilla autoencodert """
import tensorflow.keras as K


def autoencoder(input_dims, hidden_layers, latent_dims):
    """ function that contructs a vanilla autoencoder

        Parameters
        ----------
        input_dims : int
            Dimensions of the model input.
        hidden_layers : list
            List containing the number of nodes for each hidden layer in
            the encoder.
        latent_dims : int
            Dimensions of the latent space representation.

        Returns
        -------
        encoder : tensorflow.keras.Model
            The encoder model.
        decoder : tensorflow.keras.Model
            The decoder model.
        autoencoder : tensorflow.keras.Model
            The full autoencoder model.
    """

    layers = len(hidden_layers)
    decode_layers = hidden_layers[::-1]

    # encoder
    encode_inpt = K.layers.Input(shape=(input_dims,))
    layer = encode_inpt
    for i in range(layers):
        layer = K.layers.Dense(units=hidden_layers[i],
                               activation='relu')(layer)
    latent = K.layers.Dense(units=latent_dims, activation='relu')(layer)
    encode = K.Model(inputs=encode_inpt, outputs=latent)

    # decoder
    inpt = K.layers.Input(shape=(latent_dims,))
    layer = inpt
    for i in range(layers):
        layer = K.layers.Dense(units=decode_layers[i],
                               activation='relu')(layer)
    latent = K.layers.Dense(units=input_dims, activation='sigmoid')(layer)
    decode = K.Model(inputs=inpt, outputs=latent)

    # full autoencoder
    autoencode = K.Model(inputs=encode_inpt, outputs=decode(encode
                                                            (encode_inpt)))

    # compile
    autoencode.compile(optimizer='adam', loss='binary_crossentropy')

    return encode, decode, autoencode
