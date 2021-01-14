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
        new_path = save_path + ".meta"
        saver = tf.train.import_meta_graph(new_path)
        saver.restore(sess, save_path)
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        accuracy = tf.get_collection("accuracy")
        loss = tf.get_collection("loss")
        train_op = tf.get_collection("train_op")
        iterations = X_train.shape[0] // batch_size
        over_size = False
        if iterations % batch_size != 0:
            iterations += 1
            over_size = True
        for epoch in range(epochs + 1):
            a_train, l_train = sess.run([accuracy, loss],
                                        {x: X_train, y: Y_train})
            a_valid, l_valid = sess.run([accuracy, loss],
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
