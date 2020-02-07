import numpy as np
import os
import datetime
import cv2
import torch
import glob
# import roboschool, pybullet_envs, gym
from ..utils import utils
from tqdm import tqdm
from tensorboardX import SummaryWriter
from scipy import signal
from ..methods import ATD3, ATD3_RNN, Average_TD3, DDPG, \
    TD3, SAC, DDPG_RNN, TD3_RNN, ATD3_IM, SAAC


class Solver(object):
    def __init__(self, args, env, project_path):
        args.seed += args.ini_seed
        args.seed = args.seed % 10
        self.args = args
        self.env = env

        self.file_name = ''
        self.project_path = project_path
        self.result_path = project_path + "results"

        self.evaluations = []
        self.estimate_Q_vals = []
        self.true_Q_vals = []

        # Set seeds
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        state_dim = env.observation_space.shape[0]
        print('state_dim', state_dim)
        action_dim = env.action_space.shape[0]
        print('action_dim', action_dim)
        print(env.action_space.high)
        max_action = float(env.action_space.high[0])

        # Initialize policy
        if 'ATD3' == args.policy_name:
            policy = ATD3.ATD3(state_dim, action_dim, max_action)
        elif 'ATD3_IM' == args.policy_name:
            policy = ATD3_IM.ATD3_IM(state_dim, action_dim, max_action)
        elif 'ATD3_RNN' == args.policy_name:
            policy = ATD3_RNN.ATD3_RNN(state_dim, action_dim, max_action)
        elif 'DDPG_RNN' == args.policy_name:
            policy = DDPG_RNN.DDPG_RNN(state_dim, action_dim, max_action)
        elif 'TD3_RNN' == args.policy_name:
            policy = TD3_RNN.TD3_RNN(state_dim, action_dim, max_action)
        elif 'Average_TD3' == args.policy_name:
            policy = Average_TD3.Average_TD3(state_dim, action_dim, max_action, num_steps=args.average_steps)
        elif 'DDPG' == args.policy_name:
            policy = DDPG.DDPG(state_dim, action_dim, max_action)
        elif 'SAC' == args.policy_name:
            policy = SAC.SAC(state_dim, action_dim, max_action, self.env.action_space)
        elif 'SAAC' == args.policy_name:
            policy = SAAC.SAAC(state_dim, action_dim, max_action)
        else:
            policy = TD3.TD3(state_dim, action_dim, max_action)
        self.policy = policy
        self.replay_buffer = utils.ReplayBuffer()

        self.total_timesteps = 0
        self.pre_num_steps = self.total_timesteps
        self.timesteps_since_eval = 0
        self.timesteps_calc_Q_vale = 0
        self.best_reward = 0.0

    def train_once(self):
        if self.total_timesteps != 0:
            if self.args.evaluate_Q_value:
                self.writer_train.add_scalar('ave_reward', self.episode_reward, self.total_timesteps)
            self.policy.train(self.replay_buffer, self.args.batch_size, self.args.discount,
                              self.args.tau, self.args.policy_noise, self.args.noise_clip,
                              self.args.policy_freq)

    def eval_once(self):
        self.pbar.update(self.total_timesteps - self.pre_num_steps)
        self.pre_num_steps = self.total_timesteps
        if self.args.evaluate_Q_value:
            if self.total_timesteps >= self.args.start_timesteps and \
                    self.timesteps_calc_Q_vale >= self.args.eval_freq/10:
                self.timesteps_calc_Q_vale %= (self.args.eval_freq/10)
                estimate_Q_val = self.policy.cal_estimate_value(self.replay_buffer)
                self.writer_train.add_scalar('Q_value', estimate_Q_val,
                                             self.total_timesteps)
                self.estimate_Q_vals.append(estimate_Q_val)

        # Evaluate episode
        if self.timesteps_since_eval >= self.args.eval_freq:
            self.timesteps_since_eval %= self.args.eval_freq
            # evaluate the policy for once
            avg_reward = evaluate_policy(self.env, self.policy, self.args)
            self.evaluations.append(avg_reward)
            self.writer_test.add_scalar('ave_reward', avg_reward, self.total_timesteps)

            if self.args.save_all_policy:
                self.policy.save(
                    self.file_name + str(int(int(self.total_timesteps/self.args.eval_freq)* self.args.eval_freq)),
                    directory=self.log_dir)

            if self.args.evaluate_Q_value:
                true_Q_value = cal_true_value(env=self.env, policy=self.policy,
                                              replay_buffer=self.replay_buffer,
                                              args=self.args)
                self.writer_test.add_scalar('Q_value', true_Q_value, self.total_timesteps)
                self.true_Q_vals.append(true_Q_value)
                print('Estimate Q_value: {}, true Q_value: {}'.format(estimate_Q_val, true_Q_value))

            if self.best_reward < avg_reward:
                self.best_reward = avg_reward
                print("Best reward! Total T: %d Episode T: %d Reward: %f" %
                      (self.total_timesteps, self.episode_timesteps, avg_reward))
                self.policy.save(self.file_name, directory=self.log_dir)
                np.save(self.log_dir + "/test_accuracy", self.evaluations)
                utils.write_table(self.log_dir + "/test_accuracy", np.asarray(self.evaluations))
                if self.args.evaluate_Q_value:
                    utils.write_table(self.log_dir + "/estimate_Q_vals", np.asarray(self.estimate_Q_vals))
                    utils.write_table(self.log_dir + "/true_Q_vals", np.asarray(self.true_Q_vals))

    def reset(self):
        # Reset environment
        self.obs, _, done = self.env.reset()
        self.obs_vec = np.dot(np.ones((self.args.seq_len, 1)), self.obs.reshape((1, -1)))
        self.episode_reward = 0
        self.episode_timesteps = 0

    def train(self):
        self.evaluations = [evaluate_policy(self.env, self.policy, self.args)]

        if 'Average' in self.args.policy_name:
            self.log_dir = '{}/{}/{}_{}_{}_seed_{}'.format(self.result_path, self.args.log_path,
                                                        self.args.policy_name, self.args.average_steps, self.args.env_name,
                                                        self.args.seed)
        else:
            self.log_dir = '{}/{}/{}_{}_seed_{}'.format(self.result_path, self.args.log_path,
                                                        self.args.policy_name, self.args.env_name,
                                                        self.args.seed)

        print("---------------------------------------")
        print("Settings: %s" % self.log_dir)
        print("---------------------------------------")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # TesnorboardX
        if self.args.evaluate_Q_value:
            self.writer_train = SummaryWriter(logdir=self.log_dir + '_train')
        self.writer_test = SummaryWriter(logdir=self.log_dir)
        self.pbar = tqdm(total=self.args.max_timesteps, initial=self.total_timesteps, position=0, leave=True)
        done = False
        safe_or_not = True
        self.reset()
        while self.total_timesteps < self.args.max_timesteps:
            self.train_once()
            if done or not safe_or_not or self.episode_timesteps + 1 > self.args.max_episode_steps:
                print('safe_or_not', safe_or_not)
                print('done', done)
                print('total_timesteps', self.total_timesteps)
                print('episode_reward', self.episode_reward)
                self.eval_once()
                self.reset()
                done = False
                safe_or_not = True

            # Select action randomly or according to policy
            if self.total_timesteps < self.args.start_timesteps:
                action = self.env.action_space.sample()
            else:
                if 'RNN' in self.args.policy_name:
                    action = self.policy.select_action(np.array(self.obs_vec))
                elif 'SAC' in self.args.policy_name:
                    action = self.policy.select_action(np.array(self.obs), eval=False)
                else:
                    action = self.policy.select_action(np.array(self.obs))

                if self.args.expl_noise != 0:
                    action = (action + np.random.normal(0, self.args.expl_noise,
                                                        size=self.env.action_space.shape[0])).clip(
                        self.env.action_space.low[0], self.env.action_space.high[0])

            if 'IM' in self.args.policy_name:
                action_im = np.copy(action)
                action = utils.calc_torque_from_impedance(action_im,
                                                          np.asarray(self.obs)[8:-2]).clip(
                        self.env.action_space.low, self.env.action_space.high)

            new_obs, _, reward, done, safe_or_not = self.env.step(action)
            # new_obs, reward, done, _ = self.env.step(action)

            self.episode_reward += reward

            done_bool = 0 if self.episode_timesteps + 1 == self.args.max_episode_steps else float(done)

            if 'IM' in self.args.policy_name:
                action = action_im

            if 'RNN' in self.args.policy_name:
                # Store data in replay buffer
                new_obs_vec = utils.fifo_data(np.copy(self.obs_vec), new_obs)
                self.replay_buffer.add((np.copy(self.obs_vec), new_obs_vec, action, reward, done_bool))
                self.obs_vec = utils.fifo_data(self.obs_vec, new_obs)
            else:
                self.replay_buffer.add((self.obs, new_obs, action, reward, done_bool))

            self.obs = new_obs
            self.episode_timesteps += 1
            self.total_timesteps += 1
            self.timesteps_since_eval += 1
            self.timesteps_calc_Q_vale += 1

        # Final evaluation
        avg_reward = evaluate_policy(self.env, self.policy, self.args)
        self.evaluations.append(avg_reward)
        print('evaluations', self.evaluations)

        if self.best_reward < avg_reward:
            self.best_reward = avg_reward
            print("Best reward! Total T: %d Episode T: %d Reward: %f" %
                  (self.total_timesteps, self.episode_timesteps, avg_reward))
            self.policy.save(self.file_name, directory=self.log_dir)

        if self.args.save_all_policy:
            self.policy.save(self.file_name + str(int(self.args.max_timesteps)), directory=self.log_dir)

        np.save(self.log_dir + "/test_accuracy", self.evaluations)
        # print('save_dir', self.log_dir + "/test_accuracy")
        utils.write_table(self.log_dir + "/test_accuracy", np.asarray(self.evaluations))
        if self.args.evaluate_Q_value:
            true_Q_value = cal_true_value(env=self.env, policy=self.policy,
                                          replay_buffer=self.replay_buffer,
                                          args=self.args)
            self.writer_test.add_scalar('Q_value', true_Q_value, self.total_timesteps)
            self.true_Q_vals.append(true_Q_value)
            utils.write_table(self.log_dir + "/estimate_Q_vals", np.asarray(self.estimate_Q_vals))
            utils.write_table(self.log_dir + "/true_Q_vals", np.asarray(self.true_Q_vals))
        self.env.reset()

    def eval_only(self, is_reset=True):
        video_dir = '{}/video_all/{}_{}'.format(self.result_path, self.args.policy_name,
                                                self.args.env_name)
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        model_path_vec = glob.glob(self.result_path + '/{}/{}_{}_seed*'.format(
            self.args.log_path, self.args.policy_name, self.args.env_name))
        print(model_path_vec)
        for model_path in model_path_vec:
            # print(model_path)
            self.policy.load("%s" % (self.file_name + self.args.load_policy_idx), directory=model_path)
            for _ in range(1):
                video_name = video_dir + '/{}_{}_{}.mp4'.format(
                    datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                    self.file_name, self.args.load_policy_idx)
                if self.args.save_video:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out_video = cv2.VideoWriter(video_name, fourcc, 60.0, self.args.video_size)
                obs = self.env.reset()
                # print(self.env.step(np.asarray([0, 0, 0, 0, 0, 0])))
                if 'RNN' in self.args.policy_name:
                    obs_vec = np.dot(np.ones((self.args.seq_len, 1)), obs.reshape((1, -1)))

                obs_mat = np.asarray(obs)
                done = False

                while not done:
                    if 'RNN' in self.args.policy_name:
                        action = self.policy.select_action(np.array(obs_vec))
                    else:
                        action = self.policy.select_action(np.array(obs))

                    if 'IM' in self.args.policy_name:
                        action_im = np.copy(action)
                        action = utils.calc_torque_from_impedance(action_im, np.asarray(obs)[8:-2])

                    obs, reward, done, _ = self.env.step(action)

                    if 'RNN' in self.args.policy_name:
                        obs_vec = utils.fifo_data(obs_vec, obs)

                    if 0 != self.args.state_noise:
                        obs[8:20] += np.random.normal(0, self.args.state_noise, size=obs[8:20].shape[0]).clip(
                            -1, 1)

                    obs_mat = np.c_[obs_mat, np.asarray(obs)]

                    if self.args.save_video:
                        img = self.env.render(mode='rgb_array')
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        out_video.write(img)
                    elif self.args.render:
                        self.env.render(mode='human')

                if not self.args.render:
                    utils.write_table(video_name + '_state', np.transpose(obs_mat))
                if self.args.save_video:
                    out_video.release()
        if is_reset:
            self.env.reset()


# Runs policy for X episodes and returns average reward
def evaluate_policy(env, policy, args, eval_episodes=1):
    print('Evaluate untrained policy')
    avg_reward = 0.
    for _ in range(eval_episodes):
        print('eval_episodes', eval_episodes)
        obs, state, done = env.reset()
        if 'RNN' in args.policy_name:
            obs_vec = np.dot(np.ones((args.seq_len, 1)), obs.reshape((1, -1)))
        done = False
        safe_or_not = True
        eval_episodes_steps = 0
        while not done and safe_or_not and eval_episodes_steps < args.max_episode_steps:
            if 'RNN' in args.policy_name:
                action = policy.select_action(np.array(obs_vec))
            else:
                action = policy.select_action(np.array(obs))

            if 'IM' in args.policy_name:
                action_im = np.copy(action)
                action = utils.calc_torque_from_impedance(action_im, np.asarray(obs)[8:-2])
            obs, _, reward, done, safe_or_not = env.step(action)
            # obs, reward, done, _ = env.step(action)
            if 'RNN' in args.policy_name:
                obs_vec = utils.fifo_data(obs_vec, obs)
            avg_reward += reward
            eval_episodes_steps += 1
    avg_reward /= eval_episodes
    print('eval_avg_reward', avg_reward)
    return avg_reward


def cal_true_value(env, policy, replay_buffer, args, eval_episodes=1000):
    true_Q_val_vec = []
    init_state_vec, _, _, _, _ = replay_buffer.sample(eval_episodes)
    for i in range(eval_episodes):
        env.reset()
        if 'RNN' in args.policy_name:
            obs, obs_error = env.set_robot(init_state_vec[i][-1])
            obs_vec = np.copy(init_state_vec[i])
            obs_vec[-1] = np.copy(obs)
        else:
            obs, obs_error = env.set_robot(init_state_vec[i])
        true_Q_value = 0.
        if obs_error > 1e-3:
            print('Error of resetting robot: {},\n input obs: {},\n output obs: {}'.format(
                obs_error, init_state_vec[i], obs))
            continue
        done = False
        dis_gamma = 1.
        while not done:
            if 'RNN' in args.policy_name:
                action = policy.select_action(np.array(obs_vec))
            else:
                action = policy.select_action(np.array(obs))

            if 'IM' in args.policy_name:
                action_im = np.copy(action)
                action = utils.calc_torque_from_impedance(action_im, np.asarray(obs)[8:-2])

            # action = np.zeros(6, dtype=float)
            obs, reward, done, _ = env.step(action)
            true_Q_value += dis_gamma * reward
            dis_gamma *= args.discount
            if 'RNN' in args.policy_name:
                obs_vec = utils.fifo_data(obs_vec, obs)
            # env.render()
        true_Q_val_vec.append(true_Q_value)
    return np.mean(np.asarray(true_Q_val_vec))