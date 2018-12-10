#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

'''
Multi-layers Perceptrons
'''

def Multi_layer_perceptron1(x, bn_decay=0.999, is_training=True):
    # Expected input dims: b * n * 3 * 1
    conv1 = tf.contrib.layers.conv2d(x, 64, [1,3], stride=1, padding='VALID')
    bn1 = tf.contrib.layers.batch_norm(conv1, decay=bn_decay, is_training=is_training)

    conv2 = tf.contrib.layers.conv2d(bn1, 64, [1,1], stride=1, padding='VALID')
    bn2 = tf.contrib.layers.batch_norm(conv2, decay=bn_decay, is_training=is_training)

    return bn2

def Multi_layer_perceptron2(x, bn_decay=0.999, is_training=True):
    # Expected inputs dims: b * n * 1 * 64
    conv1 = tf.contrib.layers.conv2d(x, 64, [1,1], stride=1, padding='VALID')
    bn1 = tf.contrib.layers.batch_norm(conv1, decay=bn_decay, is_training=is_training)

    conv2 = tf.contrib.layers.conv2d(bn1, 128, [1,1], stride=1, padding='VALID')
    bn2 = tf.contrib.layers.batch_norm(conv2, decay=bn_decay, is_training=is_training)

    conv3 = tf.contrib.layers.conv2d(bn2, 1024, [1,1], stride=1, padding='VALID')
    bn3 = tf.contrib.layers.batch_norm(conv3, decay=bn_decay, is_training=is_training)

    return bn3

def Multi_layer_perceptron3(x, n_classes, bn_decay=0.999, is_training=True):
    # Expected input dims: b * 1024
    fc1 = tf.contrib.layers.fully_connected(x, 512, activation_fn=tf.nn.relu)
    fc_bn1 = tf.contrib.layers.batch_norm(fc1, decay=bn_decay, is_training=is_training)
    dropout1 = tf.contrib.layers.dropout(fc_bn1, keep_prob=0.7, is_training=is_training)

    fc2 = tf.contrib.layers.fully_connected(dropout1, 256, activation_fn=tf.nn.relu)
    fc_bn2 = tf.contrib.layers.batch_norm(fc2, decay=bn_decay, is_training=is_training)
    dropout2 = tf.contrib.layers.dropout(fc_bn2, keep_prob=0.7, is_training=is_training)

    fc3 = tf.contrib.layers.fully_connected(dropout2, n_classes, activation_fn=None) # dims: b * n_classes
    return fc3

'''
Input Transform
'''

def t_mlp1(x, bn_decay=0.999, is_training=True):
    # Expected inputs dims: b * n * 3 * 1
    conv1 = tf.contrib.layers.conv2d(x, 64, [1,3], stride=1, padding='VALID')
    bn1 = tf.contrib.layers.batch_norm(conv1, decay=bn_decay, is_training=is_training)

    conv2 = tf.contrib.layers.conv2d(bn1, 128, [1,1], stride=1, padding='VALID')
    bn2 = tf.contrib.layers.batch_norm(conv2, decay=bn_decay, is_training=is_training)

    conv3 = tf.contrib.layers.conv2d(bn2, 1024, [1,1], stride=1, padding='VALID')
    bn3 = tf.contrib.layers.batch_norm(conv3, decay=bn_decay, is_training=is_training) # dims: b * n * 1 * 1024

    return bn3

def input_transform(x, n_pts, bn_decay=0.999, is_training=True):
    # Expected inputs dims: b * n * 3
    mlp = t_mlp1(tf.expand_dims(x, axis=-1)) # dims: b * n * 1 * 1024

    pool = tf.contrib.layers.max_pool2d(mlp, [n_pts,1], stride=1)
    pool_flat = tf.contrib.layers.flatten(pool) # dims: b * 1024

    fc1 = tf.contrib.layers.fully_connected(pool_flat, 512, activation_fn=tf.nn.relu)
    fc_bn1 = tf.contrib.layers.batch_norm(fc1, decay=bn_decay, is_training=is_training)

    fc2 = tf.contrib.layers.fully_connected(fc_bn1, 256, activation_fn=tf.nn.relu)
    fc_bn2 = tf.contrib.layers.batch_norm(fc2, decay=bn_decay, is_training=is_training)

    fc3 = tf.contrib.layers.fully_connected(fc_bn2, 9, activation_fn=None,
                                            weights_initializer=tf.zeros_initializer(),
                                            biases_initializer=tf.constant_initializer([1, 0, 0, 0, 1, 0, 0, 0, 1]))

    fc3_mat = tf.reshape(fc3, [-1, 3, 3]) # dims: b * 3 * 3
    transform = tf.matmul(x, fc3_mat)
    return tf.expand_dims(transform, axis=-1)

'''
Feature Transform
'''

def t_mlp2(x, bn_decay=0.999, is_training=True):
    # Expected inputs dims: b * n * 1 * 64
    conv1 = tf.contrib.layers.conv2d(x, 64, [1,1], stride=1, padding='VALID')
    bn1 = tf.contrib.layers.batch_norm(conv1, decay=bn_decay, is_training=is_training)

    conv2 = tf.contrib.layers.conv2d(bn1, 128, [1,1], stride=1, padding='VALID')
    bn2 = tf.contrib.layers.batch_norm(conv2, decay=bn_decay, is_training=is_training)

    conv3 = tf.contrib.layers.conv2d(bn2, 1024, [1,1], stride=1, padding='VALID')
    bn3 = tf.contrib.layers.batch_norm(conv3, decay=bn_decay, is_training=is_training) # dims: b * n * 1 * 1024

    return bn3

def feature_transform(x, n_pts, bn_decay=0.999, is_training=True):
    # Expected inputs dims: b * n * 1 * 64
    mlp = t_mlp2(x)

    pool = tf.contrib.layers.max_pool2d(mlp, [n_pts,1], stride=1)
    pool_flat = tf.contrib.layers.flatten(pool)

    fc1 = tf.contrib.layers.fully_connected(pool_flat, 512, activation_fn=tf.nn.relu)
    fc_bn1 = tf.contrib.layers.batch_norm(fc1, decay=bn_decay, is_training=is_training)

    fc2 = tf.contrib.layers.fully_connected(fc_bn1, 256, activation_fn=tf.nn.relu)
    fc_bn2 = tf.contrib.layers.batch_norm(fc2, decay=bn_decay, is_training=is_training)

    fc3 = tf.contrib.layers.fully_connected(fc_bn2, 64*64, activation_fn=None,
                                           weights_initializer=tf.zeros_initializer(),
                                           biases_initializer=tf.constant_initializer(np.eye(8).reshape(64)))

    fc3_mat = tf.reshape(fc3, [-1, 64, 64]) # dims: b * 64 * 64
    transform = tf.matmul(tf.squeeze(x, axis=2), fc3_mat)
    return tf.expand_dims(transform, axis=2)

'''
Point Net Main
'''

def PointNet(x, n_pts, n_classes, bn_decay=0.999, is_training=True):
    # Expected inputs dims: b * n * 3

    with tf.variable_scope('input_transform'):
        transform1 = input_transform(x, n_pts, bn_decay, is_training)

    with tf.variable_scope('mlp1'):
        mlp1 = Multi_layer_perceptron1(transform1, bn_decay, is_training)

    with tf.variable_scope('feature_transform'):
        transform2 = feature_transform(mlp1, n_pts, bn_decay, is_training)

    with tf.variable_scope('mlp2'):
        mlp2 = Multi_layer_perceptron2(transform2, bn_decay, is_training)

    with tf.variable_scope('mlp3'):
        maxpool = tf.contrib.layers.max_pool2d(mlp2, [n_pts,1], stride=1)
        mlp3 = Multi_layer_perceptron3(tf.reshape(maxpool, [-1, 1024]), n_classes, bn_decay, is_training)
    return mlp3

def PointNet_vanilla(x, n_pts, n_classes, bn_decay=0.999, is_training=True):
    # Expected inputs dims: b * n * 3

    x_expanded = tf.expand_dims(x, axis=-1)

    with tf.variable_scope('vanilla_mlp1'):
        mlp1 = Multi_layer_perceptron1(x_expanded, bn_decay, is_training)

    with tf.variable_scope('vanilla_mlp2'):
        mlp2 = Multi_layer_perceptron2(mlp1, bn_decay, is_training)

    with tf.variable_scope('vanilla_mlp3'):
        maxpool = tf.contrib.layers.max_pool2d(mlp2, [n_pts,1], stride=1)
        mlp3 = Multi_layer_perceptron3(tf.reshape(maxpool, [-1, 1024]), n_classes, bn_decay, is_training)
    return mlp3
