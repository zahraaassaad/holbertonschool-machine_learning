#!/usr/bin/env python3
"""builds the inception network as described in Going Deeper
   with Convolutions (2014)"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """the keras model"""
    inputs = K.Input(shape=(224, 224, 3))
    conv1 = K.layers.Conv2D(filters=64,
                            kernel_size=(7, 7),
                            activation='relu', padding="same",
                            kernel_initializer="he_normal",
                            strides=(2, 2))(inputs)
    pool1 = K.layers.MaxPool2D(pool_size=(3, 3),
                               strides=(2, 2),
                               padding="same")(conv1)
    convr = K.layers.Conv2D(filters=64,
                            kernel_size=(1, 1),
                            activation='relu', padding="same",
                            kernel_initializer="he_normal",
                            strides=(1, 1))(pool1)
    conv2 = K.layers.Conv2D(filters=192,
                            kernel_size=(3, 3),
                            activation='relu', padding="same",
                            kernel_initializer="he_normal",
                            strides=(1, 1))(convr)
    pool2 = K.layers.MaxPool2D(pool_size=(3, 3),
                               strides=(2, 2),
                               padding="same")(conv2)
    inc_blk_3a = inception_block(pool2, [64, 96, 128, 16, 32, 32])
    inc_blk_3b = inception_block(inc_blk_3a, [128, 128, 192, 32, 96, 64])
    pool3 = K.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                               padding="same")(inc_blk_3b)
    inc_blk_4a = inception_block(pool3, [192, 96, 208, 16, 48, 64])
    inc_blk_4b = inception_block(inc_blk_4a, [160, 112, 224, 24, 64, 64])
    inc_blk_4c = inception_block(inc_blk_4b, [128, 128, 256, 24, 64, 64])
    inc_blk_4d = inception_block(inc_blk_4c, [112, 144, 288, 32, 64, 64])
    inc_blk_4e = inception_block(inc_blk_4d, [256, 160, 320, 32, 128, 128])
    pool4 = K.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                               padding="same")(inc_blk_4e)
    inc_blk_5a = inception_block(pool4, [256, 160, 320, 32, 128, 128])
    inc_blk_5b = inception_block(inc_blk_5a, [384, 192, 384, 48, 128, 128])
    pool5 = K.layers.AveragePooling2D(pool_size=(7, 7),
                                      strides=(1, 1))(inc_blk_5b)
    drop_out = K.layers.Dropout(0.4)(pool5)
    lin = K.layers.Dense(units=1000, activation='softmax',
                         kernel_initializer="he_normal")(drop_out)
    return K.Model(inputs=inputs, outputs=lin)
