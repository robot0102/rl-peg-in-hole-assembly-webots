from envs.env import ArmEnv
from algorithms.pd.PD import PD
from algorithms.pd.pd_controller import learn
import numpy as np
import os

# set env
env = ArmEnv()

# parameters
algorithm_name = 'td3'
data_path = './Data/'
num_peg = 'single'
model_path = './model/' + algorithm_name + "/"

"""parameters for running"""
nb_epochs = 5
nb_epoch_cycles = 200
nb_rollout_steps = 200

file_name = 'epochs_' + str(nb_epochs)\
            + "_episodes_" + str(nb_epoch_cycles) + \
            "_rollout_steps_" + str(nb_rollout_steps)

if not os.path.exists(data_path + algorithm_name + "/" + num_peg + "/"):
    os.makedirs(data_path + algorithm_name + "/" + num_peg + "/")

# path to store data
data_path_reward = data_path + algorithm_name + "/" + num_peg + "/" + file_name + 'reward'
data_path_steps = data_path + algorithm_name + "/" + num_peg + "/" + file_name + 'steps'
data_path_states = data_path + algorithm_name + "/" + num_peg + "/" + file_name + 'states'
data_path_times = data_path + algorithm_name + "/" + num_peg + "/" + file_name + 'times'
data_path_success_rate = data_path + algorithm_name + "/" + num_peg + "/" + file_name + 'success_rate'

model_name = file_name + 'model'
steps = []


def train():
    if algorithm_name == 'ddpg':
        from algorithms.ddpg.ddpg import learn
        learn(network='mlp',
              env=env,
              noise_type='normal_0.2',
              restore=False,
              nb_epochs=nb_epochs,
              nb_epoch_cycles=nb_epoch_cycles,
              nb_train_steps=60,
              nb_rollout_steps=nb_rollout_steps,
              data_path_reward=data_path_reward,
              data_path_steps=data_path_steps,
              data_path_states=data_path_states,
              data_path_times=data_path_times,
              model_path=model_path,
              model_name=model_name,
              )

    if algorithm_name == 'pd':
        from algorithms.pd.pd_controller import learn
        learn(
            controller=PD,
            env=env,
            nb_epochs=nb_epochs,
            nb_epoch_cycles=nb_epoch_cycles,
            nb_rollout_steps=nb_rollout_steps,
            data_path_reward=data_path_reward,
            data_path_steps=data_path_steps,
            data_path_states=data_path_states,
            data_path_times=data_path_times,
        )

    if algorithm_name == 'td3':
        from stable_baselines.td3 import TD3
        from stable_baselines.td3.policies import FeedForwardPolicy
        from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

        # Custom MLP policy with two layers
        class CustomTD3Policy(FeedForwardPolicy):
            def __init__(self, *args, **kwargs):
                super(CustomTD3Policy, self).__init__(*args, **kwargs,
                                                      layers=[64, 64],
                                                      layer_norm=False,
                                                      feature_extraction="mlp")

        scale = 0.1
        # The noise objects for TD3
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=scale * np.ones(n_actions))
        model = TD3(CustomTD3Policy, env, action_noise=action_noise, verbose=1)

        # Train the agent
        model.learn_episode(
              nb_epochs=nb_epochs,
              nb_episodes=nb_epoch_cycles,
              nb_rollout_steps=nb_rollout_steps,
              data_path_reward=data_path_reward,
              data_path_steps=data_path_steps,
              data_path_states=data_path_states,
              data_path_times=data_path_times,
              data_path_success_rate=data_path_success_rate,
              )


if __name__ == '__main__':
    train()
