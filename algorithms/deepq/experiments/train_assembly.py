# -*- coding: utf-8 -*-
"""
# @Time    : 23/10/18 8:03 PM
# @Author  : ZHIMIN HOU
# @FileName: train_assembly.py
# @Software: PyCharm
# @Github    ： https://github.com/hzm2016
"""

from baselines import deepq
from baselines.common import models
from baselines.deepq.assembly.Env_robot_control import env_search_control


def main():
    env = env_search_control()
    act = deepq.learn(
        env,
        network=models.mlp(num_hidden=32, num_layers=2),
        lr=1e-3,
        total_timesteps=5000,
        total_episodes=100,
        total_steps=50,
        target_network_update_freq=20,
        buffer_size=32,
        learning_starts=32,
        learning_times=10,
        exploration_fraction=0.2,
        exploration_final_eps=0.01,
        print_freq=10,
        param_noise=False,
        save_path='_fuzzy_noise_final_test_2'
        # load_path='assembly_model_fuzzy_final_test.pkl'
    )
    # load_path = 'assembly_model_fuzzy.pkl'
    # load_path = 'assembly_model.pkl'
    # print("Saving model to assembly_fuzzy_noise.pkl")
    act.save("./exp_second/assembly_model_fuzzy_final_test_2.pkl")


if __name__ == '__main__':
    main()