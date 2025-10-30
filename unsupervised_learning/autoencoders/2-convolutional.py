#!/usr/bin/env python3
""" tbw """
import tensorflow.keras as K


def autoencoder(input_dims, filters, latent_dims):
    """ tbw """
    layers = len(filters)
    decode_layers = filters[::-1]

    # encoder
    encode_inpt = K.layers.Input(shape=input_dims)
    layer = encode_inpt
    for i in range(layers):
        layer = K.layers.Conv2D(
            filters[i], (3, 3), padding='same', activation='relu')(layer)
        layer = K.layers.MaxPooling2D((2, 2), padding='same')(layer)
    latent = layer
    encode = K.Model(inputs=encode_inpt, outputs=latent)

    # decoder
    inpt = K.layers.Input(shape=latent_dims)
    layer = inpt
    for i in range(layers):
        if i != layers - 1:
            layer = K.layers.Conv2D(
                decode_layers[i], (3, 3), padding='same',
                activation='relu')(layer)
            layer = K.layers.UpSampling2D((2, 2))(layer)
        else:
            layer = K.layers.Conv2D(
                decode_layers[i], (3, 3), padding='valid',
                activation='relu')(layer)
            layer = K.layers.UpSampling2D((2, 2))(layer)
    latent = K.layers.Conv2D(
        input_dims[2], (3, 3), activation='sigmoid', padding='same')(layer)
    decode = K.Model(inputs=inpt, outputs=latent)

    # full autoencoder
    autoencode = K.Model(inputs=encode_inpt, outputs=decode(encode
                                                            (encode_inpt)))

    # compile
    autoencode.compile(optimizer='adam', loss='binary_crossentropy')

    return encode, decode, autoencode
