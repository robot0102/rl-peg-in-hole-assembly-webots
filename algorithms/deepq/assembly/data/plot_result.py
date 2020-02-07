# -*- coding: utf-8 -*-
"""
# @Time    : 24/10/18 2:40 PM
# @Author  : ZHIMIN HOU
# @FileName: plot_result.py
# @Software: PyCharm
# @Github    ： https://github.com/hzm2016
"""
import collections
import matplotlib.pyplot as plt
import numpy as np
import pickle
import copy as cp
from baselines.deepq.assembly.src.value_functions import *

"""=================================Plot result====================================="""
# YLABEL = ['$F_x(N)$', '$F_y(N)$', '$F_z(N)$', '$M_x(Nm)$', '$M_y(Nm)$', '$M_z(Nm)$']
YLABEL = ['$F_x$(N)', '$F_y$(N)', '$F_z$(N)', '$M_x$(Nm)', '$M_y$(Nm)', '$M_z$(Nm)']
Title = ["X axis force", "Y axis force", "Z axis force",
         "X axis moment", "Y axis moment", "Z axis moment"]
High = np.array([40, 40, 0, 5, 5, 5, 542, -36, 188, 5, 5, 5])
Low = np.array([-40, -40, -40, -5, -5, -5, 538, -42, 192, -5, -5, -5])
scale = np.array([40, 40, 40, 5, 5, 5])
"""================================================================================="""
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False


def plot(result_path):
    plt.figure(figsize=(15, 15), dpi=100)
    plt.title('Search Result')
    prediction_result = np.load(result_path)
    for i in range(len(prediction_result)):
        for j in range(6):
            line = prediction_result[:, j]
            # plt.subplot(2, 3, j+1)
            plt.plot(line)
            plt.ylabel(YLABEL[j])
            plt.xlabel('steps')
            plt.legend(YLABEL)
    plt.show()


def plot_force_and_moment(path_2, path_3):

    V_force = np.load(path_2)
    V_state = np.load(path_3)

    plt.figure(figsize=(15, 10), dpi=100)
    plt.title("Search Result of Force", fontsize=20)
    plt.plot(V_force[:100])
    plt.xlabel("Steps", fontsize=20)
    plt.ylabel("F(N)", fontsize=20)
    plt.legend(labels=['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'], loc='best', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.figure(figsize=(15, 10), dpi=100)
    plt.title("Search Result of State", fontsize=20)
    plt.plot(V_state[:100] - [539.88427, -38.68679, 190.03184, 179.88444, 1.30539, 0.21414])
    plt.xlabel("Steps", fontsize=20)
    plt.ylabel("Coordinate", fontsize=20)
    plt.legend(labels=['x', 'y', 'z', 'rx', 'ry', 'rz'], loc='best', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.show()


def plot_reward(reward_path):
    reward = np.load(reward_path)
    print(reward[0])
    plt.figure(figsize=(15, 15), dpi=100)
    plt.title('Episode Reward')
    plt.plot(np.arange(len(reward) - 1), np.array(reward[1:]))
    plt.ylabel('Episode Reward')
    plt.xlabel('Episodes')
    plt.show()


def plot_raw_data(path_1):
    data = np.load(path_1)
    force_m = np.zeros((len(data), 12))

    plt.figure(figsize=(20, 20), dpi=100)
    plt.tight_layout(pad=3, w_pad=0.5, h_pad=1.0)
    plt.subplots_adjust(left=0.065, bottom=0.1, right=0.995, top=0.9, wspace=0.2, hspace=0.2)
    plt.title("True Data")
    for j in range(len(data)):
        force_m[j] = data[j, 0]
    k = -1
    for i in range(len(data)):
        if data[i, 1] == 0:
            print("===========================================")
            line = force_m[k+1:i+1]
            print(line)
            k = i
            for j in range(6):
                plt.subplot(2, 3, j + 1)
                plt.plot(line[:, j])
                # plt.plot(line[:, 0])

                if j == 1:
                    plt.ylabel(YLABEL[j], fontsize=17.5)
                    plt.xlabel('steps', fontsize=20)
                    plt.xticks(fontsize=15)
                    plt.yticks(fontsize=15)
                else:
                    plt.ylabel(YLABEL[j], fontsize=20)
                    plt.xlabel('steps', fontsize=20)
                    plt.xticks(fontsize=15)
                    plt.yticks(fontsize=15)
        i += 1


def plot_continuous_data(path):
    raw_data = np.load(path)
    plt.figure(figsize=(20, 15))
    plt.title('Episode Reward')
    plt.tight_layout(pad=3, w_pad=0.5, h_pad=1.0)
    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.98, top=0.9, wspace=0.23, hspace=0.22)
    # plt.subplots_adjust(left=0.065, bottom=0.1, right=0.995, top=0.9, wspace=0.2, hspace=0.2)
    data = np.zeros((len(raw_data), 12))
    for j in range(len(raw_data)):
        data[j] = raw_data[j, 0]
    for j in range(6):
        plt.subplot(2, 3, j + 1)
        plt.plot(data[:, j]*scale[j], linewidth=2.5)
        # plt.ylabel(YLABEL[j], fontsize=18)
        if j>2:
            plt.xlabel('steps', fontsize=30)
        plt.title(YLABEL[j], fontsize=30)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
    # plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.2)
    plt.savefig('raw_data.pdf')
    plt.show()


def compute_true_return(path):
    raw_data = np.load(path)
    # print(raw_data)
    clock = 0
    G = 0.
    past_gammas = []
    past_cumulants = []
    all_G = []
    for i in range(len(raw_data)):
        observation, action, done, action_probability = raw_data[i]

        if done == False:
            gamma = 0.99
        else:
            gamma = 0.

        past_gammas.append(gamma)
        past_cumulants.append(1)

        if done == False:
            clock += 1
            G = 0
            all_G.append(cp.deepcopy(G))
        else:
            print('clock', clock)
            for j in reversed(range(0, clock + 1)):
                G *= past_gammas[j]
                G += past_cumulants[j]
            all_G.append(cp.deepcopy(G))
            clock = 0
            past_cumulants = []
            past_gammas = []

    print(len(raw_data))
    plt.figure(figsize=(20, 15))
    plt.plot(all_G[300:400])
    plt.show()
    return all_G


# Plot the true prediction and true value
def plot_different_gamma_data(path):
    f = open(path, 'rb')
    titles = ['$\gamma = 0.4$', '$\gamma = 0.8$', '$\gamma = 0.96$', '$\gamma =1.0$']
    # true_data = compute_true_return('prediction_result_different_gamma.npy')
    # f = open('../data/learning_result_policy', 'rb')
    # plot_value_functions = ['Move down Fy', 'Move down Fx', 'Move down Fz', 'Move down Mx', 'Move down My', 'Move down Mz']
    plot_value_functions = ['Move down step', 'Move down step 2', 'Move down step 3', 'Move down step 4']
    # plot_value_functions = ['Move down Fx', 'Move down Fx 1', 'Move down Fx 2', 'Move down Fx 3']
    raw_data = pickle.load(f)
    plt.figure(figsize=(20, 15))
    plt.tight_layout(pad=3, w_pad=1., h_pad=0.5)
    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.98, top=0.9, wspace=0.23, hspace=0.23)
    # legend = sorted([key for key in plot_value_functions.keys()])
    # print(legend)
    # print(value_functions.keys())
    for j, key in enumerate(plot_value_functions):
        plt.subplot(2, 2, j + 1)
        # print(list(raw_data[('GTD(1)', 'Hindsight Error')][key]))
        # plt.plot(np.array(raw_data[('GTD(1)', 'Hindsight Error')][key])[:], linewidth=2.5)
        # plt.plot(true_data[300:])
        plt.plot(np.array(raw_data[('GTD(0)', 'UDE')][key])[600:], linewidth=2.75)
        # print('true value', np.array(raw_data[('GTD(0)', 'UDE')][key])[300:400])
        # plt.plot(np.array(raw_data[('GTD(0)', 'TD Error')][key])[600:], linewidth=2.5)
        # print('old prediction', np.array(raw_data[('GTD(0)', 'TD Error')][key])[300:400])
        plt.plot(np.array(raw_data[('GTD(0)', 'Prediction')][key])[600:], linewidth=2.75)
        # plt.plot(np.array(raw_data[('GTD(1)', 'Prediction')][key])[300:] - np.array(raw_data[('GTD(1)', 'Hindsight Error')][key])[300:], linewidth=2.5)
        # plt.legend('True value', 'Prediction value')
        plt.title(titles[j], fontsize=30)
        if j > 1:
            plt.xlabel('steps', fontsize=30)
        plt.ylabel('Number of steps', fontsize=30)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
    # plt.savefig('different_gamma.pdf')
    plt.show()


# Plot the true prediction and true value
def chinese_plot_different_gamma_data(path):
    f = open(path, 'rb')
    titles = ['$\gamma = 0.4$', '$\gamma = 0.8$', '$\gamma = 0.96$', '$\gamma =1.0$']
    # true_data = compute_true_return('prediction_result_different_gamma.npy')
    # f = open('../data/learning_result_policy', 'rb')
    # plot_value_functions = ['Move down Fy', 'Move down Fx', 'Move down Fz', 'Move down Mx', 'Move down My', 'Move down Mz']
    plot_value_functions = ['Move down step', 'Move down step 2', 'Move down step 3', 'Move down step 4']
    # plot_value_functions = ['Move down Fx', 'Move down Fx 1', 'Move down Fx 2', 'Move down Fx 3']
    raw_data = pickle.load(f)
    plt.figure(figsize=(20, 12), dpi=1000)
    plt.tight_layout(pad=3, w_pad=1., h_pad=0.5)
    plt.subplots_adjust(left=0.08, bottom=0.12, right=0.98, top=0.95, wspace=0.23, hspace=0.33)
    # legend = sorted([key for key in plot_value_functions.keys()])
    # print(legend)
    # print(value_functions.keys())

    for j, key in enumerate(plot_value_functions):
        plt.subplot(2, 2, j + 1)
        # print(list(raw_data[('GTD(1)', 'Hindsight Error')][key]))
        # plt.plot(np.array(raw_data[('GTD(1)', 'Hindsight Error')][key])[:], linewidth=2.5)
        # plt.plot(true_data[300:])
        plt.plot(np.array(raw_data[('GTD(0)', 'UDE')][key])[600:], linewidth=2.75)
        # print('true value', np.array(raw_data[('GTD(0)', 'UDE')][key])[300:400])
        # plt.plot(np.array(raw_data[('GTD(0)', 'TD Error')][key])[600:], linewidth=2.5)
        # print('old prediction', np.array(raw_data[('GTD(0)', 'TD Error')][key])[300:400])
        plt.plot(np.array(raw_data[('GTD(0)', 'Prediction')][key])[600:], linewidth=2.75)
        # plt.plot(np.array(raw_data[('GTD(1)', 'Prediction')][key])[300:] - np.array(raw_data[('GTD(1)', 'Hindsight Error')][key])[300:], linewidth=2.5)
        # plt.legend('True value', 'Prediction value')

        plt.title(titles[j], fontsize=36)

        if j > 1:
            plt.xlabel('搜索步数', fontsize=36)
        plt.ylabel('预测周期', fontsize=36)
        plt.xticks([0, 50, 100, 150, 200], fontsize=36)
        plt.yticks(fontsize=36)

    plt.savefig('./figure/pdf/chinese_different_gamma.pdf')
    # plt.show()


def chinese_plot_compare_raw_data(path1, path2):
    raw_data = np.load(path1)
    raw_data_1 = np.load(path2)
    plt.figure(figsize=(20, 12), dpi=1000)
    plt.title('Episode Reward')
    plt.tight_layout(pad=3, w_pad=0.5, h_pad=1.0)
    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.98, top=0.95, wspace=0.33, hspace=0.15)
    data = np.zeros((len(raw_data), 12))
    for j in range(len(raw_data)):
        data[j] = raw_data[j, 0]

    data_1 = np.zeros((len(raw_data_1), 12))
    for j in range(len(raw_data_1)):
        data_1[j] = raw_data_1[j, 0]

    for j in range(6):
        plt.subplot(2, 3, j + 1)
        plt.plot(data[:100, j], linewidth=2.5, color='r', linestyle='--')
        plt.plot(data_1[:100, j], linewidth=2.5, color='b')
        # plt.ylabel(YLABEL[j], fontsize=18)
        if j>2:
            plt.xlabel('搜索步数', fontsize=38)
        plt.title(YLABEL[j], fontsize=38)
        plt.xticks(fontsize=38)
        plt.yticks(fontsize=38)

    # plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.2)
    plt.savefig('./figure/pdf/chinese_raw_data.pdf')
    # plt.show()


# Plot the true prediction and true value
def chinese_plot_different_policy_data(path, name):

    f = open(path, 'rb')
    # true_data = compute_true_return('prediction_result_different_gamma.npy')
    # f = open('../data/learning_result_policy', 'rb')
    plot_value_functions = ['Move down Fx', 'Move down Fy', 'Move down Fz', 'Move down Mx', 'Move down My', 'Move down Mz']
    # plot_value_functions = ['Move down step', 'Move down step 2', 'Move down step 3', 'Move down step 4']
    # plot_value_functions = ['Move down Fx', 'Move down Fx 1', 'Move down Fx 2', 'Move down Fx 3']
    raw_data = pickle.load(f)
    plt.figure(figsize=(20, 12), dpi=1000)
    plt.title('Episode Reward')
    plt.tight_layout(pad=3, w_pad=0.5, h_pad=1.0)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.98, top=0.95, wspace=0.33, hspace=0.25)
    # plt.subplots_adjust(left=0.1, bottom=0.12, right=0.98, top=0.94, wspace=0.23, hspace=0.33)
    # legend = sorted([key for key in plot_value_functions.keys()])
    # print(legend)
    # print(value_functions.keys())
    for j, key in enumerate(plot_value_functions):

        plt.subplot(2, 3, j + 1)
        # print(list(raw_data[('GTD(1)', 'Hindsight Error')][key]))
        # plt.plot(np.array(raw_data[('GTD(1)', 'Hindsight Error')][key])[400:]*scale[j], linewidth=2.5)
        # plt.plot(true_data[300:])
        plt.plot(np.array(raw_data[('GTD(1)', 'UDE')][key])[1000:]*scale[j], linewidth=2.5)
        # print('true value', np.array(raw_data[('GTD(0)', 'UDE')][key])[300:400])
        # plt.plot(np.array(raw_data[('GTD(0)', 'TD Error')][key])[600:], linewidth=2.5, color='r')
        # print('old prediction', np.array(raw_data[('GTD(0)', 'TD Error')][key])[300:400])
        plt.plot(np.array(raw_data[('GTD(1)', 'Prediction')][key])[1000:]*scale[j], linewidth=2.5)
        # plt.plot(np.array(raw_data[('GTD(1)', 'Prediction')][key])[300:] - np.array(raw_data[('GTD(1)', 'Hindsight Error')][key])[300:], linewidth=2.5)
        # plt.legend('True value', 'Prediction value')
        plt.title(YLABEL[j], fontsize=38)
        if j > 2:
            plt.xlabel('搜索步数', fontsize=38)
        plt.xticks([0, 50, 100, 150, 200], fontsize=38)
        plt.yticks(fontsize=38)

    plt.savefig('./figure/pdf/chinese_' + name +'.pdf')
    # plt.show()


# Plot the true prediction and true value
def plot_different_policy_data(path):
    f = open(path, 'rb')
    # true_data = compute_true_return('prediction_result_different_gamma.npy')
    # f = open('../data/learning_result_policy', 'rb')
    plot_value_functions = ['Move down Fx', 'Move down Fy', 'Move down Fz', 'Move down Mx', 'Move down My', 'Move down Mz']
    # plot_value_functions = ['Move down step', 'Move down step 2', 'Move down step 3', 'Move down step 4']
    # plot_value_functions = ['Move down Fx', 'Move down Fx 1', 'Move down Fx 2', 'Move down Fx 3']
    raw_data = pickle.load(f)
    plt.figure(figsize=(20, 12), dpi=1000)
    plt.title('Episode Reward')
    plt.tight_layout(pad=3, w_pad=1.0, h_pad=1.0)
    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.98, top=0.9, wspace=0.23, hspace=0.23)
    # legend = sorted([key for key in plot_value_functions.keys()])
    # print(legend)
    # print(value_functions.keys())
    for j, key in enumerate(plot_value_functions):

        plt.subplot(2, 3, j + 1)
        # print(list(raw_data[('GTD(1)', 'Hindsight Error')][key]))
        # plt.plot(np.array(raw_data[('GTD(1)', 'Hindsight Error')][key])[400:]*scale[j], linewidth=2.5)
        # plt.plot(true_data[300:])
        plt.plot(np.array(raw_data[('GTD(1)', 'UDE')][key])[1000:]*scale[j], linewidth=2.5)
        # print('true value', np.array(raw_data[('GTD(0)', 'UDE')][key])[300:400])
        # plt.plot(np.array(raw_data[('GTD(0)', 'TD Error')][key])[600:], linewidth=2.5, color='r')
        # print('old prediction', np.array(raw_data[('GTD(0)', 'TD Error')][key])[300:400])
        plt.plot(np.array(raw_data[('GTD(1)', 'Prediction')][key])[1000:]*scale[j], linewidth=2.5)
        # plt.plot(np.array(raw_data[('GTD(1)', 'Prediction')][key])[300:] - np.array(raw_data[('GTD(1)', 'Hindsight Error')][key])[300:], linewidth=2.5)
        # plt.legend('True value', 'Prediction value')

        plt.title(YLABEL[j], fontsize=30)
        if j > 2:
            plt.xlabel('steps', fontsize=30)
        plt.xticks([0, 50, 100, 150, 200], fontsize=25)
        plt.yticks(fontsize=25)

    plt.savefig('./figure/pdf/chinese_different_policies_b.pdf')
    # plt.show()


if __name__ == "__main__":
    # force = np.load('./search_force.npy')
    # state = np.load('./search_state.npy')
    # print(np.max(force, axis=0))
    # print(np.min(force, axis=0))
    # print(np.max(state, axis=0))
    # print(np.min(state, axis=0))
    # plot('./search_state.npy')
    # plot('./search_force.npy')
    # plot_reward('./episode_rewards.npy')

    # data = np.load('prediction_result.npy')
    # print(data[:, 2])
    # plot_continuous_data('prediction_result_different_gamma_six_force.npy')

    # f = open('../data/learning_result', 'rb')
    # y = pickle.load(f)
    # data = y[('GTD(1)', 'Hindsight Error')]['Move down Fz']
    # print(data)
    # plt.figure(figsize=(15, 15), dpi=100)
    # plt.title('Search Result')
    #
    # plt.plot(data)
    # plt.ylabel(YLABEL[0])
    # plt.xlabel('steps')
    # plt.legend(YLABEL)
    # plt.show()

    # compute_true_return('prediction_result_different_gamma.npy')
    # plot_true_data('learning_result_six_force_gamma_0.9')
    # plot_true_data('learning_result_different_gamma')
    # plot_different_gamma_data('learning_result_different_policy')

    """=============================== plot different policy ===================================== """
    # plot_different_policy_data('learning_result_six_force_gamma_0.9')
    # chinese_plot_different_policy_data('learning_result_six_force_gamma_0.9')
    # plot_different_policy_data('learning_result_different_policy_new_3')
    chinese_plot_different_policy_data('learning_result_different_policy_new_3', 'off_policy_3')
    # chinese_plot_different_policy_data('learning_result_different_policy')
    # chinese_plot_different_policy_data('learning_result_different_policy')

    """=============================== plot different gamma ======================================== """
    # plot_different_gamma_data('learning_result_different_gamma_new')
    # chinese_plot_different_gamma_data('learning_result_different_gamma_new')