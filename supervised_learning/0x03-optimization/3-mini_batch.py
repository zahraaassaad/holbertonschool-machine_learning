#!/usr/bin/env python3

"""trains a loaded neural network model
using mini-batch gradient descent."""

import numpy as np
import tensorflow as tf

shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train,
                     X_valid, Y_valid, batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt"):
    """Returns: the path where the model was saved."""
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sess, load_path)
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        train_op = tf.get_collection("train_op")[0]
        iterations = X_train.shape[0] // batch_size
        over_size = False
        if iterations % batch_size != 0:
            iterations += 1
            over_size = True
        for epoch in range(epochs + 1):
            l_train, a_train = sess.run([loss, accuracy],
                                        {x: X_train, y: Y_train})
            l_valid, a_valid = sess.run([loss, accuracy],
                                        {x: X_valid, y: Y_valid})
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(l_train))
            print("\tTraining Accuracy: {}".format(a_train))
            print("\tValidation Cost: {}".format(l_valid))
            print("\tValidation Accuracy: {}".format(a_valid))
            if epoch < epochs:
                x_tr, y_tr = shuffle_data(X_train, Y_train)
                for iter in range(iterations):
                    lower_bound = iter * batch_size
                    if over_size is True and iter == iterations - 1:
                        upper_bound = X_train.shape[0]
                    else:
                        upper_bound = iter * batch_size + batch_size
                    batch_x = X_train[lower_bound:upper_bound]
                    batch_y = Y_train[lower_bound:upper_bound]
                    sess.run([train_op], {x: batch_x, y: batch_y})
                    if epoch != 0 and (epoch + 1) % 100 == 0:
                        batch_co, batch_ac = sess.run([loss, accuracy],
                                                      {x: batch_x, y: batch_y})
                        print('\tStep {}:'.format(j + 1))
                        print('\t\tCost: {}'.format(batch_co))
                        print('\t\tAccuracy: {}'.format(batch_ac))
        return saver_n.save(sess, save_path)
