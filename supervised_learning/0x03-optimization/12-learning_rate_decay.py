#!/usr/bin/env python3
"""creates learning rate
   decay operation
"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """"returns learning rate decay operation"""
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        glo_step = sess.run(global_step)
        if glo_step % decay_rate == 0:
            alpha = tf.train.inverse_time_decay(
                alpha, global_step, decay_step, decay_rate, staircase=True)
            return alpha
        else:
            return alpha
