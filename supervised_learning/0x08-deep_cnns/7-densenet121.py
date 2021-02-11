#!/usr/bin/env python3
"""builds the DenseNet-121 architecture
   as described in Densely Connected
   Convolutional Networks:"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """the keras model"""
    inputs = K.Input(shape=(224, 224, 3))
    BN1 = K.layers.BatchNormalization(axis=3)(inputs)
    Relu1 = K.layers.Activation("relu")(BN1)
    conv1 = K.layers.Conv2D(filters=64,
                            kernel_size=(7, 7),
                            padding="same",
                            kernel_initializer="he_normal",
                            strides=(2, 2))(Relu1)
    pool1 = K.layers.MaxPool2D(pool_size=(3, 3),
                               strides=(2, 2),
                               padding="same")(conv1)

    DB1, NF1 = dense_block(pool1, 64, growth_rate, 6)
    TL1, NFT1 = transition_layer(DB1, NF1, compression)
    DB2, NF2 = dense_block(TL1, NFT1, growth_rate, 12)
    TL2, NFT2 = transition_layer(DB2, NF2, compression)
    DB3, NF3 = dense_block(TL2, NFT2, growth_rate, 24)
    TL3, NFT3 = transition_layer(DB3, NF3, compression)
    DB4, _ = dense_block(TL3, NFT3, growth_rate, 16)

    pool2 = K.layers.AveragePooling2D(pool_size=(7, 7),
                                      strides=(1, 1))(DB4)
    lin = K.layers.Dense(units=1000, activation='softmax',
                         kernel_initializer="he_normal")(pool2)
    return K.Model(inputs=inputs, outputs=lin)
