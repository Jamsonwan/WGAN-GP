import tensorflow as tf
import os
import numpy as np


def discriminator(images, reuse=False):
    images = tf.reshape(images, [-1, 28, 28, 3])

    with tf.variable_scope("discriminator", reuse=reuse):
        x1 = tf.layers.conv2d(images, 64, 5, 2, padding='same', kernel_initializer=tf.glorot_uniform_initializer())
        x1 = tf.nn.leaky_relu(x1)

        x2 = tf.layers.conv2d(x1, 128, 5, 2, padding='same', kernel_initializer=tf.glorot_uniform_initializer())
        x2 = tf.contrib.layers.layer_norm(x2)
        x2 = tf.nn.leaky_relu(x2)

        x3 = tf.layers.conv2d(x2, 256, 5, 2, padding='same', kernel_initializer=tf.glorot_uniform_initializer())
        x3 = tf.contrib.layer.layer_norm(x3)
        x3 = tf.nn.leaky_relu(x3)

        x_flat = tf.reshape(x3, [-1, 4*4*256])
        logits = tf.layers.dense(x_flat, 1)
    return logits


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    OUTPUT_DIM = 64 * 64 * 3

    input_real = tf.compat.v1.placeholder(tf.float32, [None, OUTPUT_DIM], name='input')