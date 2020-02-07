# -*- coding: utf-8 -*-
"""
# @Time    : 07/11/18 9:26 AM
# @Author  : ZHIMIN HOU
# @FileName: normal_control.py
# @Software: PyCharm
# @Github    ： https://github.com/hzm2016
"""

import tensorflow as tf
import zipfile
import cloudpickle
import numpy as np
import copy as cp
import argparse
from baselines.deepq.assembly.Env_robot_control import env_insert_control


env = env_insert_control()


def search(arg):
    Forces, States = [], []
    for i in range(arg['episodes']):
        pull_finish = env.pull_search_peg()
        state, pull_terminal = env.search_reset()

        # position control
        if pull_terminal:
            done = env.pos_control()
        Force, State = [], []

        # force control
        if done:
             for i in range(arg['steps']):
                 state = env.get_state()
                 Force.append(cp.deepcopy(state[:6]))
                 State.append(cp.deepcopy(state[6:]))
                 finish = env.force_control(env.refForce, state[:6], state[6:], i)
                 if finish:
                     break
             # pull_finish = env.pull_search_peg()
        Forces.append(cp.deepcopy(Force))
        States.append(cp.deepcopy(State))

    np.save('../data/normal_search_force_noise', Forces)
    np.save('../data/normal_search_state_noise', States)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, choices=['device', 'file'])
    parser.add_argument('--path', type=str)
    parser.add_argument('--alpha', type=float, default=0.005)
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--memory_size', type=int, default=3000)
    parser.add_argument('--data-file', type=str)
    parser.add_argument('--lambda', type=float, default=0.6)
    parser.add_argument('--meta_step_size', type=float, default=0.00001)  ## add new items
    parser.add_argument('--eta', type=float, default=0.01)
    parser.add_argument('--loop', type=float, default=1)
    parser.add_argument('--noplot', action='store_false', dest='plot')
    parser.add_argument('--record-file', type=str)
    parser.add_argument('--seed', type=int)
    return vars(parser.parse_args())


if __name__ == '__main__':
    arg = parse_args()
    search(arg)