# -*- coding: utf-8 -*-
"""
# @Time    : 25/10/18 2:32 PM
# @Author  : ZHIMIN HOU
# @FileName: performance_test_assembly.py
# @Software: PyCharm
# @Github    ： https://github.com/hzm2016
"""

import numpy as np
from baselines import deepq
from baselines.common import models
from baselines import logger
import copy as cp
from baselines.deepq.assembly.Env_robot_control import env_search_control


def main(
        test_episodes=20,
        test_steps=50
        ):
    env = env_search_control()
    print(env.observation_space)
    print(env.action_space)
    act = deepq.learn(
        env,
        network=models.mlp(num_layers=1, num_hidden=64),
        total_timesteps=0,
        total_episodes=0,
        total_steps=0,
        load_path="assembly_model_fuzzy_final.pkl"
    )
    episode_rewards = []
    episode_states = []
    for i in range(test_episodes):
        obs, done = env.reset()
        episode_rew = 0
        episode_obs = []
        logger.info("================== The {} episode start !!! ===================".format(i))
        for j in range(test_steps):
            obs, rew, done, _ = env.step(act(obs[None])[0], j)
            episode_rew += rew
            episode_obs.append(obs)
        episode_rewards.append(cp.deepcopy(episode_rew))
        episode_states.append(cp.deepcopy(episode_obs))
        print("Episode reward", episode_rew)

    np.save('../data/test_episode_reward_fuzzy_final_new', episode_rewards)
    np.save('../data/test_episode_state_fuzzy_final_new', episode_states)


if __name__ == '__main__':
    main(
        test_episodes=20,
        test_steps=50
    )