#!/usr/bin/env python3
"""train the model using early stopping"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """early stopping should only be performed
       if validation_data exists"""
    if validation_data:
        early_stop = K.callbacks.EarlyStopping(
            monitor='val_loss',  mode='min', patience=patience)
        early_stop = [early_stop]
    else:
        early_stop = None
    History = network.fit(
                        x=data,
                        y=labels,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=verbose,
                        shuffle=shuffle,
                        validation_data=validation_data,
                        callbacks=early_stop)
    return History
