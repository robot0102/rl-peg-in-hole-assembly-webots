# -*- coding: utf-8 -*-
"""
# @Time    : 27/12/18 12:45 PM
# @Author  : ZHIMIN HOU
# @FileName: impedance_control.py
# @Software: PyCharm
# @Github    ： https://github.com/hzm2016
"""

from baselines.deepq.assembly.Env_robot_control import env_search_control
import argparse
import copy as cp
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, choices=['device', 'file'])
    parser.add_argument('--path', type=str)
    parser.add_argument('--alpha', type=float, default=0.005)
    parser.add_argument('--episodes', type=int, default=150)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--memory_size', type=int, default=3000)
    parser.add_argument('--data-file', type=str)
    parser.add_argument('--lambda', type=float, default=0.6)
    parser.add_argument('--meta_step_size', type=float, default=0.00001)
    parser.add_argument('--eta', type=float, default=0.01)
    parser.add_argument('--loop', type=float, default=1)
    parser.add_argument('--noplot', action='store_false', dest='plot')
    parser.add_argument('--record-file', type=str)
    parser.add_argument('--seed', type=int)
    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()
    env = env_search_control()
    obs, state, _ = env.reset()

    epoch_force_pose = []
    epoch_action = []
    action = np.zeros(6)

    for i in range(args['steps']):
        next_obs, next_state, reward, done, safe_or_not, execute_action = env.step(np.array([0., 0, 0., 0., 0., 0.]), i)
        epoch_force_pose.append(cp.deepcopy(next_state))
        epoch_action.append(cp.deepcopy(execute_action))

        if done:
            env.pull_peg_up()

        print('Step::::\n', i)

    np.save('./data/impedance_controller_episode_states', epoch_force_pose)
    np.save('./data/impedance_controller_episode_actions', epoch_action)