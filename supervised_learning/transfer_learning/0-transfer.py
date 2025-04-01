#!/usr/bin/env python3
""" module containing function that uses transfer learning to construct a
model that identifies images in the CIFAR-10 dataset """
import tensorflow.keras as K
from tensorflow.keras import layers, callbacks, optimizers


def preprocess_data(X, Y):
    """ preprocesses data for training """
    inp = K.applications.inception_resnet_v2.preprocess_input(X)
    labels = K.utils.to_categorical(Y, 10)
    return inp, labels


def build_callbacks():
    """ Create and return training callbacks """
    return [
        callbacks.ModelCheckpoint('cifar10.h5', monitor='val_accuracy', save_best_only=True, mode='max'),
        callbacks.EarlyStopping(patience=6, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=3, min_lr=0.00001)
    ]


def build_model():
    """ Construct and return the transfer learning model """
    base_model = K.applications.InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(32, 32, 3))
    base_model.trainable = False

    inputs = K.Input(shape=(32, 32, 3))
    sc = layers.Lambda(lambda x: K.backend.resize_images(x, 229 // 32, 229 // 32, interpolation='bilinear'))(inputs)
    x = base_model(sc, training=False)

    # Use GlobalAveragePooling instead of Flatten for efficiency
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = K.Model(inputs, outputs)
    model.compile(optimizer=optimizers.Adam(),
                  loss=K.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def main():
    """ function that uses transfer learning to construct a model that
    identifies images in the CIFAR-10 dataset """
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()

    # Preprocess the data
    px_train, py_train = preprocess_data(x_train, y_train)
    px_test, py_test = preprocess_data(x_test, y_test)

    # Data augmentation
    data_gen = K.preprocessing.image.ImageDataGenerator(
        rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
        shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'
    )
    data_gen.fit(px_train)

    # Create callbacks
    cbs = build_callbacks()

    # Build the model
    model = build_model()

    # Train the model
    model.fit(
        data_gen.flow(px_train, py_train), epochs=50, batch_size=64,
        callbacks=cbs, shuffle=True, validation_data=(px_test, py_test)
    )


if __name__ == "__main__":
    main()
