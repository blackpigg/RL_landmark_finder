#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 23:31:43 2017

@author: wd
"""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import gym
from gym import spaces
from gym.utils import seeding

class NChainEnv(gym.Env):
  n-Chain environment

    This game presents moves along a linear chain of states, with two actions:
     0) forward, which moves along the chain but returns no reward
     1) backward, which returns to the beginning and has a small reward

    The end of the chain, however, presents a large reward, and by moving
    'forward' at the end of the chain this large reward can be repeated.

    At each action, there is a small probability that the agent 'slips' and the
    opposite transition is instead taken.

    The observed state is the current state in the chain (0 to n-1).

    This environment is described in section 6.1 of:
    A Bayesian Framework for Reinforcement Learning by Malcolm Strens (2000)
    http://ceit.aut.ac.ir/~shiry/lecture/machine-learning/papers/BRL-2000.pdf

    def __init__(self, n=5, slip=0.2, small=2, large=10):
        self.n = n
        self.slip = slip  # probability of 'slipping' an action
        self.small = small  # payout for 'backwards' action
        self.large = large  # payout at end of chain for 'forwards' action
        self.state = 0  # Start at beginning of the chain
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(self.n)
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action)
        if self.np_random.rand() < self.slip:
            action = not action  # agent slipped, reverse action taken
        if action:  # 'backwards': go back to the beginning, get small reward
            reward = self.small
            self.state = 0
        elif self.state < self.n - 1:  # 'forwards': go up along the chain
            reward = 0
            self.state += 1
        else:  # 'forwards': stay at the end of the chain, collect large reward
            reward = self.large
        done = False
        return self.state, reward, done, {}

    def _reset(self):
        self.state = 0
        return self.state

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import numpy as np
import gym
from gym import utils
from gym import spaces
import h5py
import tensorflow as tf
import random
import scipy.misc


class NDEnv(gym.Env):
    # ======================================================================================================================
    # load image, label and define scale of env space
    def __init__(self, desc=None, img_index=random.randint(0, 9995), length=28):

        self.imdb = h5py.File('/home/wd/Downloads/imdb_100_2.mat')
        self.image = self.imdb.get('imdb/images/data')
        self.label = self.imdb.get('imdb/images/labels')

        self.image = np.asarray(self.image)
        # normalize image
        self.image = (self.image - 127.5) / 127.5
        self.label = np.asarray(self.label)

        # define scale of agent
        self.width = length
        self.height = length

        # define scale of environment
        self.scale_factor = 1.3
        self.nrow = np.shape(self.image[0])[1] * self.scale_factor
        self.ncol = np.shape(self.image[0])[1] * self.scale_factor

        # define number of action: up, down, left, right, trigger
        self.action_space = spaces.Discrete(5)
        self._reset()

    # =======================================================================================================================
    # initialize environment and agent
    def _reset(self):

        # index of image at dataset
        self.idx = random.randint(0, 9995)
        self.desc = self.image[self.idx, :, :, :]
        # rotate image
        self.desc = np.moveaxis(self.desc, 1, 2)
        # resize image
        self.desc = scipy.misc.imresize(self.desc[0], (130, 130))

        # get label of image
        self.eye_label = self.label[self.idx, 0:2]
        self.nose_label = self.label[self.idx, 4:6]

        # scaling label
        self.gt_x = self.eye_label[0] * self.scale_factor
        self.gt_y = self.eye_label[1] * self.scale_factor
        self.gt = [self.gt_x, self.gt_y]

        gt_box_scale = 0.5
        self.gt_box = [self.gt_x - gt_box_scale * self.width, self.gt_y - gt_box_scale * self.height,
                       self.gt_x + gt_box_scale * self.width, self.gt_y + gt_box_scale * self.height]

        # initialize point of agent
        # self.x, self.y are upper-left point of agent
        self.x = np.random.randint(low=max(0, self.gt_x - 1.5 * gt_box_scale * self.width - 13.5), \
                                   high=min(self.gt_x + 1.5 * gt_box_scale * self.width - 13.5, self.nrow - self.width))

        self.y = np.random.randint(low=max(0, self.gt_y - 1.5 * gt_box_scale * self.width - 13.5), \
                                   high=min(self.gt_y + 1.5 * gt_box_scale * self.width - 13.5, self.nrow - self.width))

        # state_pt : center point of agent
        self.state_pt = [self.x + 13.5, self.y + 13.5]
        self.state_pt = np.asarray(self.state_pt)
        self.state = np.reshape(self.desc[self.x:self.x + self.width, self.y:self.y + self.height], (1, 784))

        return self.state, self.state_pt, self.gt, self.desc

    # =======================================================================================================================
    # define how state move
    def step(self, action):

        # trigger: [1,0,0,0,0]
        # right: [0,1,0,0,0]
        # down: [0,0,1,0,0]
        # left: [0,0,0,1,0]
        # up: [0,0,0,0,1]

        new_state = self.state
        new_state_pt = self.state_pt

        if (action == [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]).all():  # up
            self.y = max(self.y - 1, 0)
            new_state = self.desc[self.x:self.x + self.width, self.y:self.y + self.height]
            new_state_pt = [self.x + 13.5, self.y + 13.5]
            new_state_pt = np.asarray(new_state_pt)
        elif (action == [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).all()  # super up
            self.y = max(self.y - 10, 0)
            new_state = self.desc[self.x:self.x + self.width, self.y:self.y + self.height]
            new_state_pt = [self.x + 13.5, self.y + 13.5]
            new_state_pt = np.asarray(new_state_pt)
        elif (action == [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]).all():  # right
            self.x = min(self.x + 1, self.nrow - self.width - 2)
            new_state = self.desc[self.x:self.x + self.width, self.y:self.y + self.height]
            new_state_pt = [self.x + 13.5, self.y + 13.5]
            new_state_pt = np.asarray(new_state_pt)
        elif (action == [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]).all():  # super right
            self.x = min(self.x + 10, self.nrow - self.width - 2)
            new_state = self.desc[self.x:self.x + self.width, self.y:self.y + self.height]
            new_state_pt = [self.x + 13.5, self.y + 13.5]
            new_state_pt = np.asarray(new_state_pt)

        elif (action == [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]).all():  # down
            self.y = min(self.y + 1, self.ncol - self.height - 2)
            new_state = self.desc[self.x:self.x + self.width, self.y:self.y + self.height]
            new_state_pt = [self.x + 13.5, self.y + 13.5]
            new_state_pt = np.asarray(new_state_pt)

        elif (action == [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]).all():  # super down
            self.y = min(self.y + 10, self.ncol - self.height - 2)
            new_state = self.desc[self.x:self.x + self.width, self.y:self.y + self.height]
            new_state_pt = [self.x + 13.5, self.y + 13.5]
            new_state_pt = np.asarray(new_state_pt)

        elif (action == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]).all():  # left
            self.x = max(self.x - 1, 0)
            new_state = self.desc[self.x:self.x + self.width, self.y:self.y + self.height]
            new_state_pt = [self.x + 13.5, self.y + 13.5]
            new_state_pt = np.asarray(new_state_pt)

        elif (action == [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]).all():  # super left
            self.x = max(self.x - 10, 0)
            new_state = self.desc[self.x:self.x + self.width, self.y:self.y + self.height]
            new_state_pt = [self.x + 13.5, self.y + 13.5]
            new_state_pt = np.asarray(new_state_pt)

        elif (action == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).all():
            new_state = self.desc[self.x:self.x + self.width, self.y:self.y + self.height]
            new_state_pt = [self.x + 13.5, self.y + 13.5]
            new_state_pt = np.asarray(new_state_pt)

        new_state_pt_box = [new_state_pt[0] - 0.5 * self.width, new_state_pt[1] - 0.5 * self.height,
                            new_state_pt[0] + 0.5 * self.width, new_state_pt[1] + 0.5 * self.height]

        if (action == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).all() and self._get_IoU(self.gt_box, new_state_pt_box) > 0.7:
            done = 1
            reward = 1
        # if  (action == [1,0,0,0,0]).all() and self._inGT(self.gt_box, new_state_pt_box)==1:
        #            done = 1
        #            reward = 1
        #        if  (action == [1,0,0,0,0]).all() and self._get_distance(self.gt_box, new_state_pt_box)<2:
        #            done = 1
        #            reward = 1


        elif (action == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).all() and self._get_IoU(self.gt_box, new_state_pt_box) < 0.7:
            done = 1
            reward = -1
        # elif (action == [1,0,0,0,0]).all() and self._inGT(self.gt_box, new_state_pt_box)!=1:
        #            done = 1
        #            reward = -1
        #        elif (action == [1,0,0,0,0]).all():
        #            done = 1
        #            reward = -1


        else:
            reward = 0
            done = 0

        new_state = np.reshape(new_state, (1, 784))
        return new_state, reward, done, new_state_pt

    # =======================================================================================================================

    # get IoU of two box
    def _get_IoU(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x1 > x2 and y1 > y2:
            interArea = -1 * (x2 - x1 + 1) * (y2 - y1 + 1)
        else:
            interArea = (x2 - x1 + 1) * (y2 - y1 + 1)

        box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

        if interArea < 0:
            iou = interArea / float(box1Area + box2Area)
        else:
            iou = interArea / float(box1Area + box2Area - interArea)
        return iou

    # return binary value : value is 1 if  box2 is subset of box1
    #                      else value is 0
    def _inGT(self, box1, box2):
        x1 = box1[0] - box2[0]
        y1 = box1[1] - box2[1]
        x2 = box1[2] - box2[2]
        y2 = box1[3] - box2[3]

        if x1 < 0 and y1 < 0 and x2 > 0 and y2 > 0:
            result = 1
        else:
            result = 0
        return result

        # return distance of boxes

    def _get_distance(self, box1, box2):
        dis = np.sqrt((box1[0] - box2[0]) ** 2 + (box1[1] - box2[1]) ** 2)
        return dis


temp = NDEnv(gym.Env)
