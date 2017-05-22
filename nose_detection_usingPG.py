#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 17:02:08 2017

@author: wd
"""

import numpy as np
import tensorflow as tf
import random
import itertools
import PG_conv
import PG_value
import collections
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gym
# import dqn
# from dqn_conv import DQN
# import dqn_conv

env = gym.make('ND-v0')

input_size = 784
output_size = env.action_space.n  # number of actions

dis = 0.9
REPLAY_MEMORY = 1000


# Make batch of data and present value of reward
def replay_train(mainDQN, targetDQN, train_batch):
    # mainDQN: ....
    # targetDQN: ....
    # train_batch: queue contains training information during batch learning. Refer 'for' statement at below.
    x_stack = np.empty(0).reshape(0, 784)
    h_stack = np.empty(0).reshape(0, 50)
    y_stack = np.empty(0).reshape(0, 1)
    
    # Get stored information from the buffer
    for state, action, reward, next_state, done, history_buffer in train_batch:
        if done == 1:
            Q = reward
        else:
            Q = reward + dis*np.mean(targetDQN.predict(next_state, history_buffer))
        
        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack([x_stack, np.reshape(state, 784)])
        h_stack = np.vstack([h_stack, np.reshape(history_buffer, 50)])
    return mainDQN.update(x_stack, y_stack, h_stack)


# Convert scalar value to one-hot encoded vector
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


# 
def bot_play(i, mainDQN, success):
    # See our trained network in action
    state, state_pt, _ = env.reset()
    reward_sum = 0
    done = 0
    step_count = 0
    while not done:
        # env.render()
        a = np.argmax(mainDQN.predict(state))
        next_state, action, reward, done = env.step(a)
        reward_sum += reward
        state = next_state
        step_count += 1
        
        if 1 == done and reward == 1:
            success += 1
#            print("Total score: {}".format(reward_sum))
#            print("episode:{}, steps:{}, reward:{}, success_rate:{}".format(i, step_count, reward, success/10))
            
            break
        elif 1 == done and reward < 1:
            # print("episode:{}, steps:{}, reward:{}, success_rate:{}".format(i, step_count, reward, success/10))
            break
        elif step_count>2000:
            # print("episode:{}, steps:{}, reward:{}, success_rate:{}".format(i, step_count, reward, success/10))
            break
    return step_count, reward, success


def _get_distance(box1, box2):
        dis = np.sqrt((box1[0]-box2[0])**2+(box1[1]-box2[1])**2)
        return dis
      
# ======================================================================================================================
def main():
    weight = np.load('/home/wd/Downloads/downloads/weight_170414.npy')
    bias = np.load('/home/wd/Downloads/downloads/bias_170414.npy')
#    w3 = np.load('/home/wd/Workspace/RL/dqn_w1.npy')
#    w4 = np.load('/home/wd/Workspace/RL/dqn_w2.npy')
    cnn_eye = np.load('/home/wd/Workspace/RL/eye_cnnweight.npy')
    w3 = np.load('/home/wd/Workspace/RL/eye_weight3.npy')
    w4 = np.load('/home/wd/Workspace/RL/eye_weight4.npy')

    max_episodes = 10000

    f = open('/home/wd/Workspace/RL/print.txt','w')
    with tf.Session() as sess:
        
        # define policy network and value network
        mainDQN = PG_conv.DQN(sess, weight, w3, w4, cnn_eye, bias, input_size, output_size, 'main')
        value_estimator = PG_value.DQN(sess, weight, w3, w4, cnn_eye, bias, input_size, output_size, "target")

        tf.global_variables_initializer().run()
        
        for episode in range(max_episodes):
            done = 0
            step_count = 0
            state, state_pt, gt, whole_image = env.reset()
            replay_buffer = []
            tr = collections.namedtuple("tr", ["state", "action", "reward", "next_state", "done", "history_buffer"])
            history_buffer = deque()
            for i in range(10):
                history_buffer.append([0, 0, 0, 0, 0])
            
            for t in itertools.count():
                filename = episode
                plt.imshow(np.reshape(whole_image,(180,180)))
                ax = plt.gca()
                plt.plot(gt[0], gt[1], 'r+')
                action_pr = mainDQN.predict(state, history_buffer)
#                print(action_pr)
                tmp_action = np.random.choice(np.arange(len(action_pr)), p=action_pr)
#                print(tmp_action)
                action = convertToOneHot(np.array([tmp_action]),5)
                next_state, reward, done, new_state_pt = env.step(action[0])
                plt.plot(new_state_pt[0], new_state_pt[1], 'k.')
                plt.hold(True)

                if 1 == done and reward == 1:
                    plt.plot(state_pt[0],state_pt[1], 'b+')
                    plt.plot(new_state_pt[0],new_state_pt[1], 'ro')
                    rect = patches.Rectangle((new_state_pt[0]-13.5,new_state_pt[1]-13.5), 28, 28,
                                             linewidth=1, edgecolor='b', facecolor='none')
                    rect2 = patches.Rectangle((gt[0]-28, gt[1]-28), 56, 56,
                                              linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                    ax.add_patch(rect2)
                    plt.savefig(str(filename)+'.png')
                    print("episode:{}, gt:{}, pt: {}, step_count: {}".format(episode, gt[1], state_pt[1], step_count))
                    
                    plt.close()
                    break
                
                elif 1 == done and reward == -1: 
                    if episode % 100 == 0:
                        plt.plot(state_pt[0],state_pt[1], 'b+')
                        plt.plot(new_state_pt[0], new_state_pt[1], 'bo')
                        rect = patches.Rectangle((new_state_pt[0]-13.5,new_state_pt[1]-13.5), 28, 28,
                                                 linewidth=1, edgecolor='b', facecolor='none')
                        rect2 = patches.Rectangle((gt[0]-28, gt[1]-28), 56, 56,
                                                  linewidth=1, edgecolor='r', facecolor='none')
                        ax.add_patch(rect)
                        ax.add_patch(rect2)
                        plt.savefig(str(filename)+'.png')
                        plt.close()
                        break
                    else:
                        plt.close()
                    break
                elif step_count > 50:
                    reward = -1 
                    plt.plot(state_pt[0], state_pt[1], 'b+')
                    plt.plot(new_state_pt[0], new_state_pt[1], 'bo')
                    rect = patches.Rectangle((new_state_pt[0]-13.5, new_state_pt[1]-13.5), 28, 28,
                                             linewidth=1, edgecolor='b', facecolor='none')
                    rect2 = patches.Rectangle((gt[0]-28, gt[1]-28), 56, 56,
                                              linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                    ax.add_patch(rect2)
                    plt.savefig(str(filename)+'.png')
                    plt.close()

                    break
                history_buffer.popleft()
                history_buffer.append(action[0])
                replay_buffer.append(tr(state, action, reward, next_state, done, history_buffer))
                state = next_state
                step_count += 1

            for t, tr in enumerate(replay_buffer):
                total_return = sum(dis**i * t.reward for i, t in enumerate(replay_buffer[t:]))
                value_estimator.update(tr.state, total_return, np.reshape(tr.history_buffer, (1, 50)))
                baseline = value_estimator.predict(tr.state, np.reshape(tr.history_buffer, (1, 50)))
                delta = total_return-baseline
                mainDQN.update(tr.state, delta, np.reshape(tr.history_buffer, (1, 50)), tr.action)

if __name__ == "__main__":
    main()
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                # continuous actor-critic
#                next_value = value_estimator.predict(next_state, np.reshape(history_buffer, (1,50)))
#                target = reward + dis*next_value
#                error = target= target - value_estimator.predict(state, np.reshape(history_buffer, (1,50)))
#                value_estimator.update(state, total_return, np.reshape(tr.history_buffer, (1,50)))
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""             
