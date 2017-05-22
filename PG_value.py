#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 12:32:10 2017

@author: wd
"""

import math
import tensorflow as tf
import numpy as np
cnn = np.load('/home/wd/Workspace/RL/dqn_cnn.npy')

#   w3, w4, cnn,
class DQN:
    
    def __init__(self, sess, weight, w3, w4, cnn2, bias, input_size, output_size, name = 'main'):
        self.sess = sess
        self.weight = weight
        self.cnn = cnn
#        print(np.shape(cnn))
        self.cnn2= cnn2

        self.bias = bias
        self.input_size = input_size
        self.output_size = output_size
        self.w3 = w3
        self.w4 = w4
#        self.w5 = w5
#        self.filter_sizes = [5,5,3,3,3,1]
        self.filter_sizes = [5,5,7,1]
        self.net_name = name
        self._build_network()
        
        
    def _build_network(self, h_size=256, l_rate=1e-3):
        with tf.variable_scope(self.net_name):           
            self._X  = tf.placeholder(tf.float32, [None, self.input_size])
            self._H = tf.placeholder(tf.float32, [None, 50])
            self._Y = tf.placeholder(dtype = tf.float32)

            self.x_tensor  = tf.reshape(self._X, [-1, 28, 28, 1])
            self.current_input = self.x_tensor
            self.cnn_save = []
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
#                    batch_mean, batch_var = tf.nn.moments(tf.nn.conv2d(
#                            self.current_input, W, strides=[1, 2, 2, 1], padding='SAME'), [0,1,2])
#                    beta = tf.Variable(tf.constant(0.0, shape = [n_output]))
#                    gamma =  tf.Variable(tf.constant(1.0, shape =  [n_output]))
                    self.cnn_save.append(W)
                    self.output = tf.nn.relu(tf.nn.max_pool(
#                          
                        tf.add(tf.nn.conv2d(
                            self.current_input, W, strides=[1, 1, 1, 1], padding='SAME'), b), ksize=[1,2,2,1],\
                                strides=[1,2,2,1],padding='SAME'))
                    self.current_input = self.output
                    self.current_intput = tf.nn.dropout(self.current_input, keep_prob = 0.7)
                elif 2 <= layer_i:
#                    batch_mean, batch_var = tf.nn.moments(tf.nn.conv2d(
#                    self.current_input, W, strides=[1, 2, 2, 1], padding='VALID'), [0,1,2])
#                    beta = tf.Variable(tf.constant(0.0, shape = [n_output]))
#                    gamma =  tf.Variable(tf.constant(1.0, shape =  [n_output])) 
                    self.cnn_save.append(W)
                    self.output = tf.nn.relu(
                        tf.add(tf.nn.conv2d(
                            self.current_input, W, strides=[1, 1, 1, 1], padding='VALID'), b))
                    self.current_input = self.output
                    self.current_input = tf.nn.dropout(self.current_input, keep_prob = 0.7)
            self.current_input = tf.reshape(self.current_input, [-1, 512])
            self.fc_input = tf.concat(1,[self.current_input, self._H])
            #First layer of weights
            self.W1 = tf.get_variable("W1", shape=[562, h_size],
                                 initializer = tf.contrib.layers.xavier_initializer())
#            layer1 = tf.nn.dropout(tf.nn.tanh(tf.matmul(self.fc_input, self.W1)), keep_prob =0.7)
            layer1 = tf.nn.relu(tf.matmul(self.fc_input, self.W1))
            
            #second layer of weights
            self.W2 = tf.get_variable("W2", shape=[h_size, 5],
                                 initializer = tf.contrib.layers.xavier_initializer())
            self._Qpred = tf.nn.dropout(tf.matmul(layer1, self.W2), keep_prob = 0.7)
            #We need to define the parts of the network needed for learning a policy
            
            self.est_value =  tf.squeeze(tf.nn.softmax(self._Qpred))
            
            #Loss function
            
            self._loss = tf.squared_difference(self.est_value, self._Y) 
            #Learning
            self._train = tf.train.AdamOptimizer(
                    learning_rate=l_rate).minimize(self._loss)
            
    def predict(self,state, history):
#        x = np.reshape(self.get_processed_state(state), [None, 256])
        x = np.reshape(state, [1,784])
#        print(history)
#        print (np.shape(np.asarray(history)))

        h = np.reshape(np.asarray(history), [1, 50])
        return self.sess.run(self.est_value, feed_dict = {self._X: x, self._H : h})
    
    def update(self, x_stack, y_stack, h_stack):
        
        return self.sess.run([self._loss, self._train], feed_dict={
                self._X: x_stack, self._Y: y_stack, self._H: h_stack})
    def save(self):
        return self.sess.run(self.W3)
    def save2(self):
        return self.sess.run(self.W4)
    def savecnn(self):
        return self.sess.run(self.cnn_save)
    def savecnn2(self):
        return self.sess.run(self.cnn_2_save)
    def save_w1(self):
        return self.sess.run(self.W1)
    def save_w2(self):
        return self.sess.run(self.W2)
