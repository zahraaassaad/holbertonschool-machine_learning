#!/usr/bin/env python3
"""script that trains
   a convolutional neural
   network to classify the
   CIFAR 10 dataset"""
import tensorflow.keras as K


def preprocess_data(X, Y):
    """preprocess data so it is accepted by ResNet50 model"""
    x_test = K.applications.resnet50.preprocess_input(X)
    y_test = K.utils.to_categorical(Y)
    return x_test, y_test


if __name__ == "__main__":

    def resize_images1(X):
        """resize data so it is accepted by ResNet50 model"""
        return K.backend.resize_images(X, 7, 7,
                                       data_format="channels_last",
                                       interpolation='bilinear')
    (X_train, Y_train), (
     X_valid, Y_valid) = K.datasets.cifar10.load_data()
    optimizer = K.optimizers.Adam(lr=0.00001)
    ResNet50_model = K.applications.ResNet50(weights='imagenet',
                                             include_top=False,
                                             input_shape=(224, 224, 3))
    ResNet50_model.trainable = False
    input = K.Input(shape=(32, 32, 3))
    lambda_1 = K.layers.Lambda(resize_images1)(input)
    x = ResNet50_model(lambda_1, training=False)
    x = K.layers.Flatten()(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Dense(128, activation='relu')(x)
    x = K.layers.Dropout(0.5)(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Dense(64, activation='relu')(x)
    x = K.layers.Dropout(0.5)(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Dense(10, activation='softmax')(x)
    model = K.Model(input, x)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy', metrics=['accuracy'])
    epochs = 20
    batch_size = 256
    xt_prep, yt_prep = preprocess_data(X_train, Y_train)
    xv_prep, yv_prep = preprocess_data(X_valid, Y_valid)
    my_callbacks = [K.callbacks.ModelCheckpoint(
                    filepath='cifar10.h5', save_best_only=True)]
    model.fit(xt_prep, yt_prep,
              batch_size=batch_size,
              validation_data=(xv_prep, yv_prep),
              epochs=epochs,
              callbacks=my_callbacks)
