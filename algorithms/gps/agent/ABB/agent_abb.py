# -*- coding: utf-8 -*-
"""
# @Time    : 23/07/19 2:21 PM
# @Author  : ZHIMIN HOU
# @FileName: agent_abb.py
# @Software: PyCharm
# @Github    ： https://github.com/hzm2016
"""

import copy
import time
import numpy as np

from baselines.deepq.assembly.Env_robot_control import env_search_control
from gps.agent.agent import Agent
from gps.agent.config import AGENT
from gps.agent.agent_utils import generate_noise, setup

ACTION = 'ACTION'
from gps.sample.sample import Sample


try:
    from gps.algorithm.policy.tf_policy import TfPolicy
except ImportError:  # user does not have tf installed.
    TfPolicy = None


class AgentABB(Agent):
    """
    This file defines an agent for the ABB Robotic environment.
    All communication between the algorithms and ROS is done through
    this class.
    """
    def __init__(self, hyperparams):
        """
        Initialize agent.
        Args:
            hyperparams: Dictionary of hyperparameters.
            init_node: Whether or not to initialize a new ROS node.
        """
        config = copy.deepcopy(AGENT)
        config.update(hyperparams)
        Agent.__init__(self, config)

        self._env = env_search_control(step_max=200, fuzzy=False, add_noise=False)

        self.x0 = self._hyperparams['x0']

        self.use_tf = False
        self.observations_stale = True

    def reset(self):
        """
        Reset the agent for a particular experiment condition.
        Args:
            condition: An index into hyperparams['reset_conditions'].
        """
        obs, state, done = self._env.reset()

        return obs, state, done

    def sample(self, policy, condition, verbose=True, save=True, noisy=True):
        """
        Reset and execute a policy and collect a sample.
        Args:
            policy: A Policy object.
            condition: Which condition setup to run.
            verbose: Unused for this agent.
            save: Whether or not to store the trial into the samples.
            noisy: Whether or not to use noise during sampling.
        Returns:
            sample: A Sample object.
        """
        # user has tf installed.
        if TfPolicy is not None:
            if isinstance(policy, TfPolicy):
                self._init_tf(policy.dU)

        obs, state, _ = self.reset()
        sensor_state = {'POS_FORCE': state}
        new_sample = self._init_sample(sensor_state)
        U = np.zeros([self.T, self.dU])

        # Generate noise
        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))

        # Sample
        for t in range(self.T-1):
            print(" ========================= Step {} =========================".format(t))
            X_t = new_sample.get_X(t=t)
            obs_t = new_sample.get_obs(t=t)
            U[t, :] = policy.act(X_t, obs_t, t, noise[t, :])
            action = np.clip(U[t, :], -1, 1) * self._env.action_high_bound
            print('gps_action:', action)

            # Execute trial.
            new_obs, next_state, r, done, safe_or_not, final_action = \
                self._env.step(action, t)

            if safe_or_not is False:
                break
            sensor_next_state = {'POS_FORCE': next_state}

            self._set_sample(new_sample, sensor_next_state, t+1)

        new_sample.set(ACTION, U)
        if save:
            self._samples[condition].append(new_sample)

        return new_sample

    def test_sample(self, policy, condition, verbose=True, save=False, noisy=False, length=200):
        """
                Reset and execute a policy and collect a sample to test the learned policy.
                Args:
                    policy: A Policy object.
                    condition: Which condition setup to run.
                    verbose: Unused for this agent.
                    save: Whether or not to store the trial into the samples.
                    noisy: Whether or not to use noise during sampling.
                Returns:
                    sample: A Sample object.
        """
        # user has tf installed.
        if TfPolicy is not None:
            if isinstance(policy, TfPolicy):
                self._init_tf(policy.dU)

        start_time = time.time()
        episode_reward = 0.

        # reset state
        obs, state, _ = self.reset()
        sensor_state = {'POS_FORCE': state}
        # new_sample = self._init_sample(sensor_state)
        new_sample = self._init_test_sample(sensor_state, length)
        U = np.zeros([length, self.dU])

        # Generate noise
        if noisy:
            noise = generate_noise(length, self.dU, self._hyperparams)
        else:
            noise = np.zeros((length, self.dU))

        # Sample
        for t in range(length - 1):
            print(" ========================= Step {} =========================".format(t))
            X_t = new_sample.get_X(t=t)
            obs_t = new_sample.get_obs(t=t)

            # print('observation:', obs_t)
            U[t, :] = policy.act(X_t, obs_t, t, noise[t, :])
            action = np.clip(U[t, :], -1, 1) * self._env.action_high_bound
            print('gps_action:', action)

            # Execute trial.
            new_obs, next_state, r, done, safe_or_not, final_action = \
                self._env.step(action, t)

            episode_reward += r
            sensor_next_state = {'POS_FORCE': next_state}
            self._set_sample(new_sample, sensor_next_state, t + 1)

            if safe_or_not is False:
                break

            if done:
                break

        end_time = time.time()
        episode_time = end_time - start_time
        new_sample.set(ACTION, U)

        if save:
            self._samples[condition].append(new_sample)

        return new_sample, episode_reward, episode_time

    def _get_new_action(self, policy, obs):
        return policy.act(None, obs, None, None)

    def _init_sample(self, b2d_X):
        """
        Construct a new sample and fill in the first time step.
        """
        sample = Sample(self)
        self._set_sample(sample, b2d_X, 0)
        return sample

    def _init_test_sample(self, b2d_X, length):
        """
        Construct a new sample and fill in the first time step.
        """
        sample = Sample(self, test=True, length=length)
        self._set_sample(sample, b2d_X, 0)
        return sample

    def _set_sample(self, sample, b2d_X, t):
        for sensor in b2d_X.keys():
            sample.set(sensor, np.array(b2d_X[sensor]), t)

    def _init_tf(self, dU):
        self.current_action_id = 1
        self.dU = dU

        self.use_tf = True
