#!/usr/bin/env python3
"""evaluates based on a existing model"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """
    returns prediction, accuracy and loss_val
    """
    new_path = save_path + ".meta"
    saver = tf.train.import_meta_graph(new_path)
    with tf.Session() as sess:
        saver.restore(sess, save_path)
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        y_pred = tf.get_collection("y_pred")
        loss = tf.get_collection("loss")
        accuracy = tf.get_collection("accuracy")
        pred, a_val, l_val = sess.run([y_pred, accuracy, loss], {x: X, y: Y})
        return (pred[0], a_val[0], l_val[0])
