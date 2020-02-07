# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 10:29:32 2019

@author: kuangen
"""
from matplotlib import cm
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
import pandas as pd
import openpyxl
import numpy as np
import glob
import cv2
import sys
# sys.path.insert(0,'../')
from code.utils.utils import *
from scipy import stats, signal


def plot_error_line(t, acc_mean_mat, acc_std_mat = None, legend_vec = None,
                    marker_vec=['+', '*', 'o', 'd', 'd', '*', '', '+', 'v', 'x'],
                    line_vec=['--', '-', ':', '-.', '-', '--', '-.', ':', '--', '-.'],
                    marker_size=5,
                    init_idx = 0, idx_step = 1):
    if acc_std_mat is None:
        acc_std_mat = 0 * acc_mean_mat
    # acc_mean_mat, acc_std_mat: rows: methods, cols: time
    color_vec = plt.cm.Dark2(np.arange(8))
    for r in range(acc_mean_mat.shape[0]):
        plt.plot(t, acc_mean_mat[r, :], linestyle=line_vec[idx_step * r + init_idx],
                 marker=marker_vec[idx_step * r + init_idx], markersize=marker_size, linewidth= 1,
                 color=color_vec[(idx_step * r + init_idx) % 8])
        plt.fill_between(t, acc_mean_mat[r, :] - acc_std_mat[r, :],
                         acc_mean_mat[r, :] + acc_std_mat[r, :], alpha=0.2,
                         color=color_vec[(idx_step * r + init_idx) % 8])
    if legend_vec is not None:
        # plt.legend(legend_vec)
        plt.legend(legend_vec, loc = 'upper left')


def read_csv_vec(file_name):
    data = np.loadtxt(open(file_name, "rb"), delimiter=",", skiprows=1)
    return data[:, -1]


def write_matrix_to_xlsx(data_mat, file_path = 'data/state_of_art_test_reward.xlsx', env_name = 'Ant',
                         index_label=['DDPG']):
    df = pd.DataFrame(data_mat)
    writer = pd.ExcelWriter(file_path, engine='openpyxl', mode='a')
    df.to_excel(writer, sheet_name=env_name, index_label=tuple(index_label), header=False)
    writer.save()
    writer.close()


def write_to_existing_table(data, file_name, sheet_name='label'):
    xl = pd.read_excel(file_name, sheet_name=None, header=0, index_col=0, dtype='object')
    xl[sheet_name].iloc[1:, :5] = data
    xl[sheet_name].iloc[1:, 5] = np.mean(data, axis=-1)
    xl[sheet_name].iloc[0, 5] = np.max(xl[sheet_name].iloc[1:, 5])
    xl[sheet_name].iloc[1:, 6] = np.std(data, axis=-1)
    xl[sheet_name].iloc[0, 6] = np.min(xl[sheet_name].iloc[1:, 6])
    print(xl[sheet_name])
    with pd.ExcelWriter(file_name, engine='xlsxwriter') as writer:
        for ws_name, df_sheet in xl.items():
            df_sheet.to_excel(writer, sheet_name=ws_name)


def plot_Q_vals(reward_name_idx = None, policy_name_vec=None, result_path ='runs/ATD3_walker2d',
                       env_name = 'RoboschoolWalker2d'):
    if reward_name_idx is None:
        reward_name_idx = [0, 9, 9, 9]
    if policy_name_vec is None:
        policy_name_vec = ['TD3', 'TD3', 'ATD3', 'ATD3_RNN']
    reward_name_vec = ['r_d', 'r_s', 'r_n', 'r_lhs', 'r_cg', 'r_gs', 'r_fr', 'r_f', 'r_gv', 'r_po']
    Q_val_mat = None
    legend_vec = []
    for r in range(len(policy_name_vec)):
        reward_str = connect_str_list(reward_name_vec[:reward_name_idx[r]+1])
        legend_vec.append(policy_name_vec[r])
        legend_vec.append('True ' + policy_name_vec[r])
        # file_names_list = [glob.glob('{}/*_{}_{}*{}_train-tag*.csv'.format(
        #     result_path, policy_name_vec[r], env_name, reward_str)),
        #     glob.glob('{}/*_{}_{}*{}-tag*.csv'.format(
        #     result_path, policy_name_vec[r], env_name, reward_str))]
        file_names_list = [glob.glob('{}/*_{}_{}*/estimate_Q_vals.xls'.format(
            result_path, policy_name_vec[r], env_name)),
            glob.glob('{}/*_{}_{}*/true_Q_vals.xls'.format(
            result_path, policy_name_vec[r], env_name))]
        for i in range(len(file_names_list)):
        # for file_name_vec in file_names_list:
            file_name_vec = file_names_list[i]
            print(file_name_vec)
            for c in range(len(file_name_vec)):
                file_name = file_name_vec[c]
                dfs = pd.read_excel(file_name)
                Q_vals = dfs.values.astype(np.float)[:, 0]
                # Q_vals = read_csv_vec(file_name)
                if Q_val_mat is None:
                    Q_val_mat = np.zeros((len(reward_name_idx) * 2, len(file_name_vec), 271))
                if Q_vals.shape[0] < Q_val_mat.shape[-1]:
                    Q_vals = np.interp(np.arange(271), np.arange(271, step = 10), Q_vals[:28])
                Q_val_mat[2 * r + i, c, :] = Q_vals[:271]

    if Q_val_mat is not None:
        fig = plt.figure(figsize=(3.5, 2.5))
        # plt.tight_layout()
        plt.rcParams.update({'font.size': 7, 'font.serif': 'Times New Roman'})
        plt.subplot(1, 2, 1)
        time_step = Q_val_mat.shape[-1] - 1
        for i in range(Q_val_mat.shape[0]):
            if 0 == i % 2:
                t = np.linspace(0, 0 + 0.01 * time_step, time_step + 1)
                plot_acc_mat(Q_val_mat[[i]],
                             None, env_name, smooth_weight=0.0, plot_std=True,
                             fig_name=None, y_label='Q value', fig = fig, t = t, marker_size = 0, init_idx=i)
            else:
                t = np.linspace(0, 0 + 0.01 * time_step, time_step / 10 + 1)
                plot_acc_mat(Q_val_mat[[i], :, ::10],
                             None, env_name, smooth_weight=0.0, plot_std=True,
                             fig_name=None, y_label='Q value', fig=fig, t=t, init_idx=i, marker_size = 2)
        plt.xlabel(r'Time steps ($1 \times 10^{5}$)')
        plt.yticks([0, 50, 100])
        plt.subplot(1, 2, 2)
        Q_val_mat = Q_val_mat[:, :, 90:]
        time_step = Q_val_mat.shape[-1] - 1
        t = np.linspace(1, 1 + 0.01 * time_step, time_step + 1)
        # error_Q_val_mat = (Q_val_mat[[0, 2]] - Q_val_mat[[1, 3]]) / Q_val_mat[[1, 3]]
        error_Q_val_mat = (Q_val_mat[0:6:2] - Q_val_mat[1:6:2]) / np.mean(Q_val_mat[1:6:2],
                                                                          axis = 1, keepdims=True)
        print('Mean absolute normalized error of Q value, TD3: {}, ATD3: {}, ATD3_RNN: {}'.format(
            np.mean(np.abs(error_Q_val_mat[0, :, -50:])), np.mean(np.abs(error_Q_val_mat[1, :, -50:])),
            np.mean(np.abs(error_Q_val_mat[2, :, -50:]))))
        plot_acc_mat(error_Q_val_mat,
                     None, env_name, smooth_weight=0.0, plot_std=True,
                     fig_name=None, y_label='Error of Q value / True Q value',
                     fig = fig, t = t, init_idx=0, idx_step=2, marker_size = 0)
        plt.xlabel(r'Time steps ($1 \times 10^{5}$)')
        plt.yticks([0, 1, 2])
        plt.xticks([1.0, 1.5, 2.0, 2.5])
        legend = fig.legend(legend_vec,
                            loc='lower center', ncol=3, bbox_to_anchor=(0.50, 0.90),
                            frameon=False)
        fig.tight_layout()
        plt.savefig('images/{}_{}.pdf'.format(env_name, 'Q_value'), bbox_inches='tight', pad_inches=0.05)
        plt.show()


def plot_error_bar(x_vec, y_mat, x_tick_vec = None):
    mean_vec = np.mean(y_mat, axis = -1)
    std_vec = np.std(y_mat, axis = -1)
    len_vec = len(x_vec)
    fig = plt.figure(figsize=(3.5, 1))
    plt.tight_layout()
    plt.rcParams.update({'font.size': 7, 'font.serif': 'Times New Roman'})

    plt.errorbar(x_vec, mean_vec, yerr = std_vec, fmt='-', elinewidth= 1,
                 solid_capstyle='projecting', capsize= 3, color = 'black')
    plt.ylabel('Test reward')
    if x_tick_vec is not None:
        plt.xticks(np.arange(len(x_tick_vec)), x_tick_vec)
    plt.savefig('images/ablation_reward.pdf', bbox_inches='tight')
    plt.show()


def connect_str_list(str_list):
    if 0 >= len(str_list):
        return ''
    str_out = str_list[0]
    for i in range(1, len(str_list)):
        str_out = str_out + '_' + str_list[i]
    return str_out


def plot_acc_mat(acc_mat, legend_vec, env_name, plot_std = True, smooth_weight = 0.8, eval_freq = 0.05,
                 t = None, fig = None, fig_name = None, y_label = 'Test reward',
                 init_idx = 0, idx_step = 1, marker_size = 2):
    # print(legend_vec)
    for r in range(acc_mat.shape[0]):
        for c in range(acc_mat.shape[1]):
            acc_mat[r, c, :] = smooth(acc_mat[r, c, :], weight=smooth_weight)
    mean_acc = np.mean(acc_mat, axis=1)
    std_acc = np.std(acc_mat, axis=1)
    # kernel = np.ones((1, 1), np.float32) / 1
    # mean_acc = cv2.filter2D(mean_acc, -1, kernel)
    # std_acc = cv2.filter2D(std_acc, -1, kernel)
    if t is None:
        time_step = acc_mat.shape[-1] - 1
        t = np.linspace(0, eval_freq * time_step, time_step+1)
    if fig is None:
        fig = plt.figure(figsize=(9, 6))
        # fig = plt.figure()
        plt.tight_layout()
        plt.rcParams.update({'font.size': 7, 'font.serif': 'Times New Roman'})
    if plot_std:
        plot_error_line(t, mean_acc, std_acc, legend_vec=legend_vec,
                        init_idx=init_idx, idx_step = idx_step, marker_size=marker_size)
    else:
        plot_error_line(t, mean_acc, legend_vec=legend_vec,
                        init_idx=init_idx, idx_step = idx_step, marker_size=marker_size)
    plt.xlabel('Time steps ' + r'($1 \times 10^{5}$)' + '\n{}'.format(env_name))
    plt.xlim((min(t), max(t)))
    plt.ylabel(y_label)
    if fig is None:
        plt.show()


def plot_reward_curves(policy_name_vec=None, result_path ='runs/ATD3_walker2d',
                       env_name = 'RoboschoolWalker2d', fig = None, fig_name='test_reward',
                       smooth_weight=0.8, eval_freq = 0.05):
    if policy_name_vec is None:
        policy_name_vec = ['TD3', 'TD3', 'ATD3', 'ATD3_RNN']
    reward_mat = None
    legend_vec = []
    last_reward = 0.0
    for r in range(len(policy_name_vec)):
        legend_vec.append(policy_name_vec[r])
        file_name_vec = glob.glob('{}/{}_{}*/test_accuracy.npy'.format(
            result_path, policy_name_vec[r], env_name))
        print(file_name_vec)
        for c in range(len(file_name_vec)):
            file_name = file_name_vec[c]
            # dfs = pd.read_excel(file_name)
            dfs = np.load(file_name)
            # acc_vec = dfs.values.astype(np.float)[:, 0]
            acc_vec = dfs
            if reward_mat is None:
                reward_mat = np.zeros((len(policy_name_vec), len(file_name_vec), len(acc_vec)))
            reward_mat[r, c, :] = acc_vec

        if reward_mat is not None:
            max_acc = np.max(reward_mat[r, :, :], axis=-1)
            # print(max_acc)
            print('Max acc for {}, mean: {}, std: {}, d_reward:{}'.format(
                policy_name_vec[r], np.mean(max_acc, axis=-1),
                np.std(max_acc, axis=-1), np.mean(max_acc, axis=-1)-last_reward))
            last_reward = np.mean(max_acc, axis=-1)

    if reward_mat is not None:
        # write_matrix_to_xlsx(np.max(reward_mat, axis = -1), env_name=env_name, index_label=policy_name_vec)
        # write_to_existing_table(np.max(reward_mat, axis = -1), file_name='data/state_of_art_test_reward_1e6.xlsx',
        #                         sheet_name=env_name)
        plot_acc_mat(reward_mat, None, env_name, fig=fig, fig_name=fig_name,
                     smooth_weight=smooth_weight, eval_freq=eval_freq, marker_size=0)
    return legend_vec


def plot_roboschool_test_reward():
    env_name_vec = [
        # 'RoboschoolHopper-v1',
        # 'RoboschoolWalker2d-v1',
        # 'RoboschoolHalfCheetah-v1',
        # 'RoboschoolAnt-v1',
        # 'Hopper-v2',
        # 'Walker2d-v2',
        # 'HalfCheetah-v2',
        # 'Ant-v2',
        # 'Peg-in-hole-single_assembly',
        '2_Friction_0.5_Clearance_1_Single_seed_1'
    ]
    fig = plt.figure(figsize=(6, 5))
    fig.tight_layout()
    plt.rcParams.update({'font.size': 8, 'font.serif': 'Times New Roman'})
    policy_name_vec = ['Average_TD3']

    for i in range(len(env_name_vec)):
        plt.subplot(2, 2, i+1)
        legend_vec = plot_reward_curves(result_path='results/runs/single_assembly',
                                        env_name=env_name_vec[i],
                                        policy_name_vec=policy_name_vec, fig=fig)
        # plt.yticks([0, 1000, 2000, 3000])
        # plt.xticks([0, 5, 10])

    print(legend_vec)
    legend = fig.legend(policy_name_vec,
                        loc='lower center', ncol=len(policy_name_vec),
                        bbox_to_anchor=(0.50, 0.96), frameon=False)
    fig.tight_layout()
    plt.savefig('Assembly.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show()


def calc_TD_reward(reward_Q):
    reward_Q_TD = np.zeros(reward_Q.shape)
    reward_Q_TD[:, 0] = reward_Q[:, 0]
    for r in range(reward_Q.shape[0]-1):
        reward_Q_TD[r, 1:3] = reward_Q[r, 1:3] - 0.99 * np.min(reward_Q[r+1, 1:3])
    return reward_Q_TD


def calc_expected_reward(reward_Q):
    reward = np.copy(reward_Q[:, 0])
    r = 0
    # for r in range(reward_Q.shape[0]-1):
    for c in range(r+1, reward_Q.shape[0]):
        reward[r] += 0.99 ** (c-r) * reward[c]
        # reward[r] += np.min(0.99 * reward_Q[r + 1, 1:3])
    init_rewar_Q = reward_Q[[0],:]
    init_rewar_Q[0, 0] = reward[0]
    return init_rewar_Q


def smooth(scalars, weight=0.9):
    # Exponential moving average,
    # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    return np.asarray(smoothed)

# # # Fig: ablation study for different rewards
# print('------Fig: ablation reward------')
# plot_ablation_reward()
#
# # Fig: test acc
# print('------Fig: test reward------')
# plot_all_test_reward()
#
# ## Fig: Q-value
# print('------Fig: Q value ------')
# plot_Q_vals(result_path = 'runs/ATD3_walker2d_Q_value',
#             env_name = 'RoboschoolWalker2d',
#             policy_name_vec = ['TD3', 'ATD3', 'ATD3_RNN'],
#             reward_name_idx = [0, 0, 0])
#
#
# # # # Fig: joint angle
# print('-----Fig: joint angle-----')
# plot_all_gait_angle()


# Fig: test acc
print('------Fig: test reward------')
plot_roboschool_test_reward()
