#!/usr/bin/env python3
"""based on 7-train also save the best iteration of the model"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False, filepath=None, verbose=True,
                shuffle=False):
    """filepath is the file path where the model should be saved"""
    call_backs = []
    if validation_data:
        early_stop = K.callbacks.EarlyStopping(
            monitor='val_loss', mode='min', patience=patience)
        call_backs.append(early_stop)
        if learning_rate_decay is True:
            def scheduler(epoch):
                return alpha / (1 + decay_rate * epoch)
            lr_decay = K.callbacks.LearningRateScheduler(
                schedule=scheduler, verbose=1)
            call_backs.append(lr_decay)
    if filepath:
        call_backs.append(K.callbacks.ModelCheckpoint(
            filepath=filepath, save_best_only=save_best))
    if len(call_backs) == 0:
        call_backs = None
    History = network.fit(
                        x=data,
                        y=labels,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=verbose,
                        shuffle=shuffle,
                        validation_data=validation_data,
                        callbacks=call_backs)
    return History
