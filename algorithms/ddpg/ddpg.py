import time

from algorithms.ddpg.ddpg_learner import DDPG
from algorithms.ddpg.models import Actor, Critic
from algorithms.ddpg.memory import Memory
from algorithms.ddpg.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
import algorithms.common.tf_util as U

from algorithms import logger

""" import your environment """
from envs.env import ArmEnv

import algorithms.common.tf_util as U

from algorithms.common import logger

import numpy as np
import copy as cp


def learn(network,
          env,
          data_path_reward="",
          data_path_steps="",
          data_path_states="",
          data_path_times="",
          model_path="",
          model_name="",
          restore=False,
          seed=None,
          nb_epochs=5,   # with default settings, perform 1M steps total
          nb_epoch_cycles=150,
          nb_rollout_steps=400,
          reward_scale=1.0,
          noise_type='normal_0.2',  #'adaptive-param_0.2',  ou_0.2, normal_0.2
          normalize_returns=False,
          normalize_observations=True,
          critic_l2_reg=1e-2,
          actor_lr=1e-4,
          critic_lr=1e-3,
          popart=False,
          gamma=0.99,
          clip_norm=None,
          nb_train_steps=50,  # per epoch cycle and MPI worker,
          batch_size=32,  # per MPI worker
          tau=0.01,
          param_noise_adaption_interval=50,
          **network_kwargs):

    nb_actions = env.action_space.shape[0]

    memory = Memory(limit=int(1e5), action_shape=env.action_space.shape[0], observation_shape=env.observation_space.shape)

    critic = Critic(network=network, **network_kwargs)
    actor = Actor(nb_actions, network=network, **network_kwargs)

    """ set noise """
    action_noise = None
    param_noise = None

    if noise_type is not None:
        for current_noise_type in noise_type.split(','):
            current_noise_type = current_noise_type.strip()
            if current_noise_type == 'none':
                pass
            elif 'adaptive-param' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
            elif 'normal' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
            elif 'ou' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
            else:
                raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    """action scale"""
    max_action = env.action_high_bound
    logger.info('scaling actions by {} before executing in env'.format(max_action))

    """ agent ddpg """
    agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape[0],
        gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm, reward_scale=reward_scale)

    logger.info('Using agent with the following configuration:')
    logger.info(str(agent.__dict__.items()))

    sess = U.get_session()

    if restore:
        agent.restore(sess, model_path, model_name)
    else:
        agent.initialize(sess)
        sess.graph.finalize()

    agent.reset()

    episodes = 0
    epochs_rewards = np.zeros((nb_epochs, nb_epoch_cycles), dtype=np.float32)
    epochs_times = np.zeros((nb_epochs, nb_epoch_cycles), dtype=np.float32)
    epochs_steps = np.zeros((nb_epochs, nb_epoch_cycles), dtype=np.float32)
    epochs_states = []
    for epoch in range(nb_epochs):

        logger.info("======================== The {} epoch start !!! =========================".format(epoch))
        epoch_episode_rewards = []
        epoch_episode_steps = []
        epoch_episode_times = []
        epoch_actions = []
        epoch_episode_states = []
        epoch_qs = []
        epoch_episodes = 0

        for cycle in range(nb_epoch_cycles):
            start_time = time.time()
            obs, state, done = env.reset()
            episode_reward = 0.
            episode_step = 0
            episode_states = []
            logger.info("================== The {} episode start !!! ===================".format(cycle))

            for t_rollout in range(nb_rollout_steps):
                # logger.info("================== The {} steps finish  !!! ===================".format(t_rollout))

                """ choose next action """
                action, q, _, _ = agent.step(obs, stddev, apply_noise=True, compute_Q=True)

                new_obs, next_state, r, done, safe_or_not = env.step(max_action * action)
                """ normalize state """
                print("\nReward", r)

                if safe_or_not is False:
                    break

                episode_reward += r
                episode_step += 1

                episode_states.append([cp.deepcopy(state), cp.deepcopy(action), np.array(cp.deepcopy(r)), cp.deepcopy(next_state)])

                epoch_actions.append(action)
                epoch_qs.append(q)

                agent.store_transition(obs, action, r, new_obs, done)
                obs = new_obs
                state = next_state

                if done:
                    break

            """ noise decay """
            stddev = float(stddev) * 0.95

            """ store data """
            duration = time.time() - start_time
            epoch_episode_rewards.append(episode_reward)
            epoch_episode_steps.append(episode_step)
            epoch_episode_times.append(cp.deepcopy(duration))
            epoch_episode_states.append(cp.deepcopy(episode_states))

            epochs_rewards[epoch, cycle] = episode_reward
            epochs_steps[epoch, cycle] = episode_step
            epochs_times[epoch, cycle] = cp.deepcopy(duration)

            logger.info("============================= The Episode_Reward:: {}!!! ============================".format(epoch_episode_rewards))
            logger.info("============================= The Episode_Times:: {}!!! ============================".format(epoch_episode_times))

            epoch_episodes += 1
            episodes += 1

            """ Training process """
            epoch_actor_losses = []
            epoch_critic_losses = []
            epoch_adaptive_distances = []
            for t_train in range(nb_train_steps):
                logger.info("")

                # Adapt param noise, if necessary.

                if memory.nb_entries >= batch_size and t_train % param_noise_adaption_interval == 0:
                    distance = agent.adapt_param_noise()
                    epoch_adaptive_distances.append(distance)
                cl, al = agent.train()
                epoch_critic_losses.append(cl)
                epoch_actor_losses.append(al)
                agent.update_target_net()

        epochs_states.append(cp.deepcopy(epoch_episode_states))

        # # save data
        np.save(data_path_reward, epochs_rewards)
        np.save(data_path_steps, epochs_steps)
        np.save(data_path_states, epochs_states)
        np.save(data_path_times, epochs_times)

        # np.save(data_path + 'train_reward_' + "DDPG" + '_' + file_name + "_" + noise_type, epochs_rewards)
        # np.save(data_path + 'train_step_' + "DDPG" + '_' + file_name + "_" + noise_type, epochs_steps)
        # np.save(data_path + 'train_states_' + "DDPG" + '_' + file_name + "_" + noise_type, epochs_states)
        # np.save(data_path + 'train_times_' + "DDPG" + '_' + file_name + "_" + noise_type, epochs_times)

    # # agent save
    agent.store(model_path + model_name)


if __name__ == '__main__':

    """ import environment """

    env = ArmEnv()
    data_path = './ddpg_data/'
    model_path = './ddpg_model/'
    nb_epochs = 5
    nb_epoch_cycles = 100
    nb_rollout_steps = 200
    file_name = '_epochs_' + str(nb_epochs) + "_episodes_" + str(nb_epoch_cycles) + "_rollout_steps_" + str(nb_rollout_steps)

    learn(network='mlp',
          env=env,
          data_path=data_path,
          noise_type='normal_0.2',
          file_name=file_name,
          model_path=model_path,
          restore=False,
          nb_epochs=nb_epochs,
          nb_epoch_cycles=nb_epoch_cycles,
          nb_rollout_steps=nb_rollout_steps,
          nb_train_steps=60,
)