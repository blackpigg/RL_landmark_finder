#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 15:49:52 2017

@author: wd
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 22:56:43 2017

@author: wd
"""
import numpy as np
import tensorflow as tf
import random
import PG_supervised_CNN_action_class
from collections import deque

import gym

env = gym.make('ND-v0')
input_size = 784
output_size = env.action_space.n # number of actions

dis = 0.9
REPLAY_MEMORY = 1000

def convertToOneHot(vector, num_classes=None):
    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0

    if num_classes is None:
        num_classes = np.max(vector)+1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result.astype(int)


def training_regnet(net, train_batch):
    x_stack = np.empty(0).reshape(0, 784)
    a_stack = np.empty(0).reshape(0, 5)
    c_stack = np.empty(0).reshape(0, 2)
    
    # Get stored information from the buffer
    for state, state_pt, gt in train_batch:
        temp = state_pt - gt
        
        if abs(temp[0]) < 1 and abs(temp[1]) < 1:
            a = [1, 0, 0, 0, 0]
            c = [1, 0]
        elif abs(temp[0]) > abs(temp[1]) and temp[0] > 0:
            a = [0, 0, 0, 1, 0]  # left
            c = [0, 1]
        elif abs(temp[0]) > abs(temp[1]) and temp[0] < 0:
            a = [0, 1, 0, 0, 0]  # right
            c = [0, 1]
        elif abs(temp[0]) < abs(temp[1]) and temp[1] > 0:
            a = [0, 0, 0, 0, 1]  # up
            c = [0, 1]
        elif abs(temp[0]) < abs(temp[1]) and temp[1] < 0:
            a = [0, 0, 1, 0, 0]  # down
            c = [0, 1]
        else:
            a = convertToOneHot(np.asarray([np.random.randint(5)]), 5)
            c = [0, 1]
            
                
        x_stack = np.vstack([x_stack, np.reshape(state, 784)])
        a_stack = np.vstack([a_stack, a])
        c_stack = np.vstack([c_stack, c])
        
    return net.update(x_stack, a_stack, c_stack)

def main():

    # f = open("sl_result.txt",'w')
    training_batch = []
    test_batch =[]
    for j in range(50):
        state2, state_pt2, gt2, _ = env.reset()
        test_batch.append((state2, state_pt2, gt2))
    for i in range(100000):
        state, state_pt, gt, _ = env.reset()
        training_batch.append((state, state_pt, gt))
#    print(np.shape(training_batch[0][0]))
    with tf.Session() as sess:
        SL_net = PG_supervised_CNN_action_class.CNN_SL(sess, input_size, output_size)
        tf.global_variables_initializer().run()
        for i in range(5):
            np.random.shuffle(training_batch)
            for j in range(99900):
                training_batch2 = training_batch[j:j+100]
                accuracy, _ = training_regnet(SL_net, training_batch2)
                if 0 == j % 100:
                    print(j, accuracy)
                    print(np.argmax((SL_net.predict(training_batch[j][0][0]))), SL_net.predict(training_batch[j][0]),
                            (training_batch[j][1] - training_batch[j][2]))
            eye_w1 = SL_net.get_wc1()
            eye_w2 = SL_net.get_wc2()
            eye_w3 = SL_net.get_wc3()
            eye_cnn = SL_net.get_cnn_weights()

            np.save('data/eye_weight1.npy', eye_w1)
            np.save('data/eye_weight2.npy', eye_w2)
            np.save('data/eye_weight3.npy', eye_w3)
            np.save('data/eye_cnnweight.npy', eye_cnn)
            
        # f.close()

if __name__=="__main__":
    main()