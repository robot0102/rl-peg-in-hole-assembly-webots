import numpy as np
import pandas as pd
import math
import torch
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy import signal


# Code based on:
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

# Expects tuples of (state, next_state, action, reward, done)
class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def get(self, idx):
        return self.storage[idx]

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def add_final_reward(self, final_reward, steps, delay=0):
        len_buffer = len(self.storage)
        for i in range(len_buffer - steps - delay, len_buffer - delay):
            item = list(self.storage[i])
            item[3] += final_reward
            self.storage[i] = tuple(item)

    def add_specific_reward(self, reward_vec, idx_vec):
        for i in range(len(idx_vec)):
            time_step_num = int(idx_vec[i])
            item = list(self.storage[time_step_num])
            item[3] += reward_vec[i]
            self.storage[time_step_num] = tuple(item)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


def read_table(file_name='../../data/joint_angle.xls', sheet_name='walk_fast'):
    dfs = pd.read_excel(file_name, sheet_name=sheet_name)
    data = dfs.values[1:-1, -6:].astype(np.float)
    return data


def write_table(file_name, data):
    df = pd.DataFrame(data)
    df.to_excel(file_name + '.xls', index=False)


def calc_gait_symmetry(joint_angle):
    joint_num = int(joint_angle.shape[-1] / 2)
    half_num_sample = int(joint_angle.shape[0] / 2)
    joint_angle_origin = np.copy(joint_angle)
    joint_angle[0:half_num_sample, joint_num:] = joint_angle_origin[half_num_sample:, joint_num:]
    joint_angle[half_num_sample:, joint_num:] = joint_angle_origin[0:half_num_sample, joint_num:]
    dist = np.zeros(joint_num)
    for c in range(joint_num):
        dist[c] = 1 - distance.cosine(joint_angle[:, c], joint_angle[:, c + joint_num])
    return np.mean(dist)


def calc_cos_similarity(joint_angle_resample, human_joint_angle):
    joint_num = human_joint_angle.shape[0]
    dist = np.zeros(joint_num)
    for c in range(joint_num):
        dist[c] = 1 - distance.cosine(joint_angle_resample[c, :], human_joint_angle[c, :])
    return np.mean(dist)


def joint_state_to_deg(joint_state_mat):
    joint_deg_mat = np.zeros(joint_state_mat.shape)
    joint_deg_mat[:, [0, 3]] = joint_state_mat[:, [0, 3]] * 80.0 + 35.0
    joint_deg_mat[:, [1, 4]] = (1 - joint_state_mat[:, [1, 4]]) * 75.0
    joint_deg_mat[:, [2, 5]] = joint_state_mat[:, [2, 5]] * 45.0
    return joint_deg_mat


def calc_array_symmetry(array_a, array_b):
    cols = array_a.shape[-1]
    dist = np.zeros(cols)
    for c in range(cols):
        dist[c] = 1 - distance.cosine(array_a[:, c], array_b[:, c])
    return np.mean(dist)


def calc_cross_gait_reward(gait_state_mat, gait_velocity, reward_name):
    """
    reward_name_vec =['r_d', 'r_s', 'r_f', 'r_n', 'r_gv', 'r_lhs', 'r_gs', 'r_cg', 'r_fr', 'r_po']
    """
    cross_gait_reward = 0.0
    reward_str_list = []
    frame_num = gait_state_mat.shape[0]
    joint_deg_mat = joint_state_to_deg(gait_state_mat[:, :-2])
    ankle_to_hip_deg_mat = joint_deg_mat[:, [0, 3]] - joint_deg_mat[:, [1, 4]]
    if 'r_gv' in reward_name:
        '''
        gait velocity
        '''
        reward_str_list.append('r_gv')
        cross_gait_reward += 0.2 * np.mean(gait_velocity)

    if 'r_lhs' in reward_name:
        '''
        0: left heel strike: the left foot should contact ground between 40% to 60% gait cycle
        Theoretical situation: 0, -1: right foot strike; 50: left foot strike
        '''
        reward_str_list.append('r_lhs')

        l_foot_contact_vec = signal.medfilt(gait_state_mat[:, -1], 3)
        l_foot_contact_vec[1:] -= l_foot_contact_vec[:-1]
        l_foot_contact_vec[0] = 0
        if 0 == np.mean(l_foot_contact_vec == 1):
            # print(gait_state_mat_sampled)
            return cross_gait_reward, reward_str_list
        l_heel_strike_idx = np.where(l_foot_contact_vec == 1)[0][0]
        cross_gait_reward += 0.2 * (1.0 - np.tanh((l_heel_strike_idx / (frame_num + 0.0) - 0.5) ** 2))


        if 'r_gs' in reward_name:
            '''
            1: gait symmetry
            '''
            reward_str_list.append('r_gs')

            r_gait_state_origin = gait_state_mat[:, np.r_[0:3, -2]]
            l_gait_state_origin = gait_state_mat[:, np.r_[3:6, -1]]
            l_gait_state = np.zeros(l_gait_state_origin.shape)
            l_gait_state[0:(frame_num - l_heel_strike_idx), :] = l_gait_state_origin[l_heel_strike_idx:, :]
            l_gait_state[(frame_num - l_heel_strike_idx):, :] = l_gait_state_origin[0:l_heel_strike_idx, :]
            cross_gait_reward += 0.2 * calc_array_symmetry(r_gait_state_origin, l_gait_state)


        if 'r_cg' in reward_name:
            '''
            2: cross gait
            '''
            reward_str_list.append('r_cg')
            cross_gait_reward += (0.2 / 4.0) * (np.tanh(ankle_to_hip_deg_mat[0, 0]) +
                                                np.tanh(- ankle_to_hip_deg_mat[l_heel_strike_idx, 0]) +
                                                # np.tanh(ankle_to_hip_deg_mat[-1, 0]) + \
                                                np.tanh(-ankle_to_hip_deg_mat[0, 1])
                                                + np.tanh(ankle_to_hip_deg_mat[l_heel_strike_idx, 1])
                                                # + np.tanh(-ankle_to_hip_deg_mat[-1, 1])
                                                )

        # if ankle_to_hip_deg_mat[0, 0] > 5 \
        #         and ankle_to_hip_deg_mat[l_heel_strike_idx, 0] < -5 \
        #         and ankle_to_hip_deg_mat[-1, 0] > 5:
        #     cross_gait_reward += 0.1
        #
        # if ankle_to_hip_deg_mat[0, 1] < -5 \
        #         and ankle_to_hip_deg_mat[l_heel_strike_idx, 1] > 5 \
        #         and ankle_to_hip_deg_mat[-1, 1] < -5:
        #     cross_gait_reward += 0.1
        if 'r_fr' in reward_name:
            '''
            3: foot recovery
            '''
            reward_str_list.append('r_fr')

            ankle_to_hip_speed_mat = np.zeros(ankle_to_hip_deg_mat.shape)
            ankle_to_hip_speed_mat[1:] = ankle_to_hip_deg_mat[1:] - ankle_to_hip_deg_mat[:-1]
            cross_gait_reward += -0.1 * (np.tanh(ankle_to_hip_speed_mat[-1, 0]) +
                                         np.tanh(ankle_to_hip_speed_mat[l_heel_strike_idx, 1]))
        if 'r_po' in reward_name:
            '''
            4: push off
            '''
            reward_str_list.append('r_po')

            r_foot_contact_vec = signal.medfilt(gait_state_mat[:, -2], 3)
            r_foot_contact_vec[1:] -= r_foot_contact_vec[:-1]
            r_foot_contact_vec[0] = 0
            ankle_speed_mat = np.zeros(joint_deg_mat[:, [2, 5]].shape)
            ankle_speed_mat[1:] = joint_deg_mat[1:, [2, 5]] - joint_deg_mat[:-1, [2, 5]]

            if 0 == np.mean(r_foot_contact_vec == -1):
                return cross_gait_reward, reward_str_list
            r_push_off_idx = np.where(r_foot_contact_vec == -1)[0][0]
            cross_gait_reward += -0.1 * np.tanh(ankle_speed_mat[r_push_off_idx, 0])

            if 0 == np.mean(l_foot_contact_vec == -1):
                return cross_gait_reward, reward_str_list
            l_push_off_idx = np.where(l_foot_contact_vec == -1)[0][0]
            cross_gait_reward += -0.1 * np.tanh(ankle_speed_mat[l_push_off_idx, 1])
    return cross_gait_reward, reward_str_list


def connect_str_list(str_list):
    if 0 >= len(str_list):
        return ''
    str_out = str_list[0]
    for i in range(1, len(str_list)):
        str_out = str_out + '_' + str_list[i]
    return str_out


def check_cross_gait(gait_state_mat):
    gait_num_1 = np.mean((gait_state_mat[:, 0] - gait_state_mat[:, 3]) > 0.1)
    gait_num_2 = np.mean((gait_state_mat[:, 0] - gait_state_mat[:, 3]) < -0.1)
    return (gait_num_1 > 0) and (gait_num_2 > 0)


def plot_joint_angle(joint_angle_resample, human_joint_angle):
    fig, axs = plt.subplots(human_joint_angle.shape[1])
    for c in range(len(axs)):
        axs[c].plot(joint_angle_resample[:, c])
        axs[c].plot(human_joint_angle[:, c])
    plt.legend(['walker 2d', 'human'])
    plt.show()


def fifo_data(data_mat, data):
    data_mat[:-1] = data_mat[1:]
    data_mat[-1] = data
    return data_mat


def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    len_mean = mean.shape
    log_z = log_std
    z = len_mean[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p


def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def calc_torque_from_impedance(action_im, joint_states, scale = 1.0):
    k_vec = action_im[0::3]
    b_vec = action_im[1::3]
    q_e_vec = action_im[2::3]
    q_vec = joint_states[0::2]
    q_v_vec = joint_states[0::2]
    action = (k_vec * (q_e_vec - q_vec) - b_vec * q_v_vec)/scale
    return action

