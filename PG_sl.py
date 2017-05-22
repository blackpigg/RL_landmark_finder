#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 13:13:26 2017

@author: wd
"""

import math
import tensorflow as tf
import numpy as np

class SL:
    
    def __init__(self, sess, input_size, output_size, name = 'main'):
        self.sess = sess
        self.input_size = input_size
        self.output_size = output_size
        self.filter_sizes = [5,5,7,1]
        self.net_name = name
        self._build_network()
        
        
    def _build_network(self, h_size=256, l_rate=1e-1):
        with tf.variable_scope(self.net_name):           
            self._X  = tf.placeholder(tf.float32, [None, self.input_size])
            self._C = tf.placeholder(tf.float32, [None, 2])
            self._A = tf.placeholder(tf.float32,[None, 5])

            self.x_tensor  = tf.reshape(self._X, [-1, 28, 28, 1])
            self.current_input = self.x_tensor
            self.cnn_weight = []
#=======================================================================================================================            
#convolution layer 
            for layer_i, n_output in enumerate([64, 128, 256 , 512]):
                n_input = self.current_input.get_shape().as_list()[3]
                W = tf.Variable(
                        tf.random_uniform([
                        self.filter_sizes[layer_i],
                        self.filter_sizes[layer_i],
                        n_input, n_output],
                        -1.0 / math.sqrt(n_input),
                        1.0 / math.sqrt(n_input)), name = 'CNN')
                b = tf.Variable(tf.zeros([n_output]))
                if 2 > layer_i:
                    self.cnn_weight.append(W)
                    self.output = tf.nn.relu(tf.nn.max_pool(
#                          
                        tf.add(tf.nn.conv2d(
                            self.current_input, W, strides=[1, 1, 1, 1], padding='SAME'), b), ksize=[1,2,2,1],\
                                strides=[1,2,2,1],padding='SAME'))
                    self.current_input = self.output
                    self.current_intput = tf.nn.dropout(self.current_input, keep_prob = 0.7)
                elif 2 <= layer_i:
                    self.cnn_weight.append(W)
                    self.output = tf.nn.relu(
                        tf.add(tf.nn.conv2d(
                            self.current_input, W, strides=[1, 1, 1, 1], padding='VALID'), b))
                    self.current_input = self.output
                    self.current_input = tf.nn.dropout(self.current_input, keep_prob = 0.7)
#=======================================================================================================================
#fully connected layer1
            self.current_input = tf.reshape(self.current_input, [-1, 512])
            #First layer of weights
            self.W1 = tf.get_variable("W1", shape=[512, h_size],
                                 initializer = tf.contrib.layers.xavier_initializer())
            
#
            layer1 = tf.nn.dropout(tf.nn.tanh(tf.matmul(self.current_input, self.W1)), keep_prob=0.7)
#=======================================================================================================================           
#action predict layer(2-1)
            self.W2 = tf.get_variable("W2", shape=[h_size, 5],
                                 initializer = tf.contrib.layers.xavier_initializer())

            self._apred = tf.matmul(layer1, self.W2)
            self.loss1 = tf.nn.softmax_cross_entropy_with_logits(logits=self._apred, labels=self._A)
#=======================================================================================================================           
#class predict layer(2-2)
            self.W3 = tf.get_variable("W3", shape=[h_size, 2], initializer=tf.contrib.layers.xavier_initializer())
            
            self._cpred = tf.matmul(layer1, self.W3)
            self.loss2 = tf.nn.softmax_cross_entropy_with_logits(logits=self._cpred, labels=self._C)
            
            #Loss function
            self._loss = tf.reduce_mean(self.loss1+self.loss2)
            #Learning
            self._train = tf.train.AdamOptimizer(
                    learning_rate=l_rate).minimize(self._loss)
            
    def predict(self,state):
        x = np.reshape(state, [1,784])
        return self.sess.run(self._apred, feed_dict = {self._X: x})

    def update(self, x_stack, a_stack, c_stack):
        # print(self.sess.run([self._loss], feed_dict={
        #         self._X: x_stack, self._A: a_stack, self._C: c_stack}))
        return self.sess.run([self._loss, self._train], feed_dict={
                self._X: x_stack, self._A: a_stack, self._C: c_stack})
    def savecnn(self):
        return self.sess.run(self.cnn_weight)
    def save_w1(self):
        return self.sess.run(self.W1)
    def save_w2(self):
        return self.sess.run(self.W2)
    def save_w3(self):
        return self.sess.run(self.W3)

