#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 13:13:26 2017

@author: wd
"""

import math
import tensorflow as tf
import numpy as np


class CNN_SL:
    
    def __init__(self, sess, input_size, output_size, name = 'main'):
        self.sess = sess
        self.input_size = input_size
        self.output_size = output_size
        self.filter_sizes = [5, 5, 7, 1]
        self.net_name = name
        self._build_network()
        
    def _build_network(self, h_size=256, l_rate=1e-5):
        with tf.variable_scope(self.net_name):           
            self._X = tf.placeholder(tf.float32, [None, self.input_size])
            self._Y = tf.placeholder(tf.float32, [None, 2])
            self._A = tf.placeholder(tf.float32, [None, 5])
            self.x_tensor = tf.reshape(self._X, [-1, 28, 28, 1])

            # ===============================================================================
            # convolution layers
            # ===============================================================================
            self.current_input = self.x_tensor
            self.cnn_weight = []
            for layer_i, nf_output in enumerate([64, 128, 256, 512]):
                nf_input = self.current_input.get_shape().as_list()[3]

                # define filters
                W = tf.Variable(tf.random_uniform(
                    [self.filter_sizes[layer_i], self.filter_sizes[layer_i], nf_input, nf_output],
                    -1.0 / math.sqrt(nf_input),
                    1.0 / math.sqrt(nf_input)),
                    name='CNN')
                b = tf.Variable(tf.zeros([nf_output]))
                self.cnn_weight.append(W)
                if 2 > layer_i:
                    filter_res = \
                        tf.nn.max_pool(
                            tf.add(tf.nn.conv2d(self.current_input, W, strides=[1, 1, 1, 1], padding='SAME'), b),
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')
                else:
                    filter_res = \
                        tf.add(tf.nn.conv2d(self.current_input, W, strides=[1, 1, 1, 1], padding='VALID'), b)
                self.current_input = tf.nn.dropout(tf.nn.relu(filter_res), keep_prob=0.7)

            # ===============================================================================
            # fully connected layers
            # ===============================================================================

            # fc layer 1
            self.current_input = tf.reshape(self.current_input, [-1, 512])
            self.Wc1 = tf.get_variable("Wc1", shape=[512, h_size], initializer=tf.contrib.layers.xavier_initializer())
            layer1 = tf.nn.dropout(tf.nn.tanh(tf.matmul(self.current_input, self.Wc1)), keep_prob=0.7)

            # fc layer for action (trigger, right, down, left, up)
            self.Wc2 = tf.get_variable("Wc2", shape=[h_size, 5], initializer=tf.contrib.layers.xavier_initializer())
            self.action_pred = tf.matmul(layer1, self.Wc2)

            # fc layer for class (landmark / not landmark)
            self.Wc3 = tf.get_variable("Wc3", shape=[h_size, 2], initializer=tf.contrib.layers.xavier_initializer())
            self.class_pred= tf.matmul(layer1, self.Wc3)

            # ===============================================================================
            # loss
            # ===============================================================================
            self.loss1 = tf.nn.softmax_cross_entropy_with_logits(logits=self.action_pred, labels=self._A)
            self.loss2 = tf.nn.softmax_cross_entropy_with_logits(logits=self.class_pred, labels=self._Y)
            self._loss = tf.reduce_mean(self.loss1 + self.loss2)

            # ===============================================================================
            # optimizer
            # ===============================================================================
            self._train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self._loss)
            
    def predict(self, state):
        x = np.reshape(state, [1, 784])
        return self.sess.run(self.action_pred, feed_dict={self._X: x})

    def update(self, input_state_batch, action_batch, class_label_batch):
        # print(self.sess.run([self._loss], feed_dict={
        #         self._X: x_stack, self._A: a_stack, self._C: c_stack}))
        return self.sess.run(
            [self._loss, self._train],
            feed_dict={self._X: input_state_batch, self._A: action_batch, self._Y: class_label_batch})

    def get_cnn_weights(self):
        return self.sess.run(self.cnn_weight)

    def get_wc1(self):
        return self.sess.run(self.Wc1)

    def get_wc2(self):
        return self.sess.run(self.Wc2)

    def get_wc3(self):
        return self.sess.run(self.Wc3)

