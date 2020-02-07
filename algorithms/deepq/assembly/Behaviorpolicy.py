# -*- coding: utf-8 -*-
"""
# @Time    : 08/09/18 4:53 PM
# @Author  : ZHIMIN HOU
# @FileName: Behaviorpolicy.py
# @Software: PyCharm
# @Github    ： https://github.com/hzm2016
"""
from random import randint
import numpy as np


class BehaviorPolicy():
    """
    0 = move_x
    1 = move_y
    2 = move_z
    3 = move_rx,
    4 = move_ry,
    5 = move_rz,
    """

    def __init__(self, policy="radom"):
        self.lastAction = 0
        self.i = 0
        self.policy_type = policy
        self.action_ability = 0

    def policy(self, state):
        if self.policy_type == "random":
            return self.randomPolicy(state)
        elif self.policy_type == "BackForth":
            return self.backAndForthPolicy(state)
        else:
            return self.fiveRightPolicy(state)

    def randomPolicy(self, state):
        actions = np.array([0, 1, 2, 3, 4, 5])
        action_prob = np.array([0.1, 0.1, 0.5, 0.1, 0.1, 0.1])

        action = np.random.choice(actions, p=action_prob)
        return action, action_prob[action]

    def backAndForthPolicy(self, state):
        if self.lastAction == 1:
            self.lastAction = 2
            return 2
        else:
            self.lastAction = 1
            return 1

    def fiveRightPolicy(self, state):
        self.i = self.i + 1
        if self.i % 5 ==0:
            if self.lastAction == 1:
                self.lastAction = 2
            else:
                self.lastAction = 1
            i = 0
        return self.lastAction