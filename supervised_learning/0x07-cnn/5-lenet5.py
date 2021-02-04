#!/usr/bin/env python3
""" Module to create LeNet-5 with Keras"""
import tensorflow.keras as K


def lenet5(X):
    """ Keras LeNet-5 """
    init = K.initializers.he_normal(seed=None)
    conv2d_1 = K.layers.Conv2D(filters=6, padding='same',
                               kernel_initializer=init,
                               kernel_size=5, activation='relu')(X)

    pool_1 = K.layers.MaxPool2D(strides=2)(conv2d_1)

    conv2d_2 = K.layers.Conv2D(filters=16, padding='valid',
                               kernel_initializer=init,
                               kernel_size=5, activation='relu')(pool_1)

    pool_2 = K.layers.MaxPool2D(strides=2)(conv2d_2)

    flat = K.layers.Flatten()(pool_2)

    dense_1 = K.layers.Dense(units=120, kernel_initializer=init,
                             activation='relu')(flat)

    dense_2 = K.layers.Dense(units=84, kernel_initializer=init,
                             activation='relu')(dense_1)

    last_dense = K.layers.Dense(units=10, kernel_initializer=init,
                                activation='softmax')(dense_2)

    model = K.models.Model(inputs=X, outputs=last_dense)

    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model
