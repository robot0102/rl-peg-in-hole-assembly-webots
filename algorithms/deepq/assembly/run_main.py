# -*- coding: utf-8 -*-
"""
# @Time    : 08/09/18 12:17 PM
# @Author  : ZHIMIN HOU
# @FileName: run_main.py
# @Software: PyCharm
# @Github    ： https://github.com/hzm2016
"""

from baselines.deepq.assembly.Env_robot_control import env_insert_control, env_search_control
from baselines.deepq.assembly.Behaviorpolicy import BehaviorPolicy
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import numpy as np
import copy as cp
import skfuzzy.control as ctrl
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting


def train(arg):
    behavior = BehaviorPolicy(policy="random")
    robot_control = env_insert_control()
    record = []
    for i in range(arg["episodes"]):
        state, pull_terminal = robot_control.search_reset()
        gamma = 1.
        while gamma > 0.:
            action = behavior.policy(state)
            next_state, gamma, safe_or_not = robot_control.step_prediction(action, gamma)
            record.append((cp.deepcopy(state),
                           cp.deepcopy(gamma),
                           cp.deepcopy(np.array(action))))
                         # cp.deepcopy(next_state)))
            state = next_state
        while pull_terminal is False:
            pull_terminal, safe_else = robot_control.pull_up()
        # record.append(cp.deepcopy(path))
        print("==================== One Interation Finished!! ===================")
    print(record)
    np.save('./data/prediction_random', record)


def search(arg):
    env = env_search_control()
    # pull_finish = env.pull_peg_up()
    # force, state, pull_terminal = env.search_reset()
    #
    # # position control
    # if pull_terminal:
    #     done = env.pos_control()
    state, obs, done = env.reset()
    print('force', state[:6])
    print('state', state[6:])

    Force, State = [], []

    # force control
    if done:
         for i in range(arg['steps']):
             current_state = env.get_state()
             force, state = current_state[:6], current_state[6:]
             Force.append(cp.deepcopy(force))
             State.append(cp.deepcopy(state))
             _, _, finish = env.step(0, i)
             if finish:
                 break
         pull_finish = env.pull_search_peg()
    np.save('../data/search_force', Force)
    np.save('../data/search_state', State)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, choices=['device', 'file'])
    parser.add_argument('--path', type=str)
    parser.add_argument('--alpha', type=float, default=0.005)
    parser.add_argument('--episodes', type=int, default=2000)
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


if __name__ == "__main__":
    arg = parse_args()
    # train(arg)
    # search(arg)
    fuzzy_control()