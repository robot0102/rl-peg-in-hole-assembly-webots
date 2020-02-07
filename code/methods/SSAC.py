import os
import torch
import torch.nn.functional as F
import glob
import numpy as np
from torch.optim import Adam
from utils.utils import soft_update, hard_update
from utils.model import GaussianPolicy, QNetwork, DeterministicPolicy
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, merge, Lambda, Activation
from keras.layers.merge import Add, Multiply, Concatenate, concatenate
from keras.initializers import RandomUniform
from keras.optimizers import Adam
import keras.backend as K
from keras import metrics


def weighted_entropy(p, w_norm):
    # w = tf.divide(tf.exp(A - np.max(A)), prob)
    # w_norm = w / K.sum(w)
    return K.sum(w_norm * p * K.log(p + 1e-8))


def weighted_mean(p, w_norm):
    # w = tf.exp(A- np.max(A))
    # w_norm = w / K.sum(w)
    p_weighted = np.multiply(w_norm, p)
    return K.mean(p_weighted, axis=0)


def weighted_mse(Q_target, Q_pred, w_norm):
    # w = tf.exp(A- np.max(A))
    # w_norm = w / K.sum(w)
    error = K.square(Q_target - Q_pred)
    return K.mean(w_norm * error)


def softmax(x):
    col = x.shape[1]
    x_max = np.reshape(np.amax(x, axis=1), (-1, 1))
    e_x = np.exp(x - np.matlib.repmat(x_max, 1, col) )
    e_x_sum = np.reshape( np.sum(e_x, axis=1), (-1, 1))
    out = e_x / np.matlib.repmat(e_x_sum, 1, col)
    return out


def weighted_mean_array(x, weights):
    weights_mean = np.mean(weights, axis=1)
    x_weighted = np.multiply(x, weights)
    mean_weighted = np.divide(np.mean(x_weighted, axis=1), weights_mean)
    return np.reshape(mean_weighted, (-1, 1))


def p_sample(p):
    row, col = p.shape
    p_sum = np.reshape(np.sum(p, axis=1), (row, 1))
    p_normalized = p/np.matlib.repmat(p_sum, 1, col)
    p_cumsum = np.matrix(np.cumsum( p_normalized, axis=1))
    # print(p_cumsum[0])
    rand = np.matlib.repmat(np.random.random((row, 1)), 1, col)
    # print(rand[0])
    o_softmax = np.argmax(p_cumsum >= rand, axis=1)
    return o_softmax


def entropy(p):
    return K.sum(p * K.log((p + 1e-8)))


def add_normal(x_input, outshape, at_eps):
    """
    add normal noise to the input
    """
    epsilon = K.random_normal(shape=outshape, mean=0., stddev=1.)
    x_out = x_input + at_eps * np.multiply(epsilon, np.absolute(x_input))
    return x_out


def kl(p, q):
    return K.sum(p * K.log((p + 1e-8) / (q + 1e-8)))


class Multi_SAC(object):
    def __init__(self, state_dim, action_dim, option_dim, max_action, action_space):

        self.alpha = 0.2
        self.lr = 0.0003
        self.option_num = option_dim

        self.policy_type = "Gaussian"
        self.target_update_interval = 1
        self.automatic_entropy_tuning = True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        """ critic network """
        self.critic = QNetwork(state_dim, action_dim, 400).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.critic_target = QNetwork(state_dim, action_dim, 400).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.sampling_prob = torch.FloatTensor(state).to(self.device)
        # ===================================================================== #
        #                              Option Model                             #
        # ===================================================================== #
        self.option_state_input, self.option_action_input, self.option_input_concat, self.option_out_dec, \
                                self.option_out, self.option_out_noise, self.option_model = self.create_option_model()
        Advantage = np.stop_gradient(self.target_q_value - self.predicted_v_value)
        Weight = np.divide(np.exp(Advantage - np.max(Advantage)), self.sampling_prob)
        W_norm = Weight/K.mean(Weight)

        critic_conditional_entropy = weighted_entropy(self.option_out, tf.stop_gradient(W_norm))
        p_weighted_ave = weighted_mean(self.option_out, tf.stop_gradient(W_norm))
        self.critic_entropy = critic_conditional_entropy - self.c_ent * entropy(p_weighted_ave)

        self.vat_loss = kl(self.option_out, self.option_out_noise)
        self.reg_loss = metrics.mean_absolute_error(self.option_input_concat, self.option_out_dec)
        self.option_loss = self.reg_loss + self.entropy_coeff * (self.critic_entropy) + self.c_reg * self.vat_loss
        self.option_optimize = tf.train.AdamOptimizer(self.option_lr).minimize(self.option_loss)

        """ option network """
        self.it = 0

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning == True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=self.lr)

            self.policy = GaussianPolicy(state_dim, action_dim, 400, max_action).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)

        elif self.policy_type == "Multi_Gaussian":
            if self.automatic_entropy_tuning == True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=self.lr)

            self.policy = GaussianPolicy(state_dim, action_dim, 400, max_action).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(state_dim, action_dim, 400, max_action).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)

    def select_action(self, state, eval=True):

        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        if eval == False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def train_actor_option(self, inputs, a_gradient, option):
        self.sess.run(self.actor_optimizer_list[option], feed_dict={
            self.actor_state_input_list[option]: inputs,
            self.action_gradient_list[option]: a_gradient
        })

    def train_critic(self, inputs, action, target_q_value, predicted_v_value, sampling_prob):
        return self.sess.run([self.critic_optimize], feed_dict={
            self.critic_state_input: inputs,
            self.critic_action_input: action,
            self.target_q_value: target_q_value,
            self.predicted_v_value: predicted_v_value,
            self.sampling_prob: sampling_prob
        })

    def train_option(self, inputs, action, target_q_value, predicted_v_value, sampling_prob):
        return self.sess.run([self.option_optimize], feed_dict={
            self.option_state_input: inputs,
            self.option_action_input: action,
            self.target_q_value: target_q_value,
            self.predicted_v_value: predicted_v_value,
            self.sampling_prob: sampling_prob
        })

    def max_option(self, inputs):
        Q_predict = []
        n = inputs.shape[0]
        for o in range(int(self.option_num)):
            action_i = self.predict_actor_target(inputs, o)
            Q_predict_i, _ = self.predict_critic_target(inputs, action_i)
            if o == 0:
                Q_predict = np.reshape(Q_predict_i, (-1, 1))
            else:
                Q_predict = np.concatenate((Q_predict, np.reshape(Q_predict_i, (-1, 1))), axis=1)

        o_max = np.argmax(Q_predict, axis=1)
        Q_max = np.max(Q_predict, axis=1)
        return o_max, Q_max, Q_predict

    def softmax_option_target(self, inputs):
        Q_predict = []
        n = inputs.shape[0]
        for o in range(int(self.option_num)):
            action_i = self.predict_actor_target(inputs, o)
            Q_predict_i, _ = self.predict_critic_target(inputs, action_i)

            if o == 0:
                Q_predict = np.reshape( Q_predict_i, (-1, 1) )
            else:
                Q_predict = np.concatenate((Q_predict, np.reshape(Q_predict_i, (-1, 1)) ), axis= 1)

        p = softmax(Q_predict)
        o_softmax = p_sample(p)
        n = Q_predict.shape[0]
        Q_softmax = Q_predict[np.arange(n), o_softmax.flatten()]

        return o_softmax, np.reshape(Q_softmax, (n, 1)), Q_predict

    def predict_actor_option(self, inputs, option):
        return self.sess.run(self.actor_out_list[option], feed_dict={self.actor_state_input_list[option]: inputs})

    def predict_actor(self, inputs, options):
        action_list = []
        for o in range(self.option_num):
            action_o = self.predict_actor_option(inputs, o)
            action_list.append(action_o)

        n = inputs.shape[0]
        action = 0
        if n == 1 or np.isscalar(options):
            action = action_list[options]
            # calculate the action
        else:
            for i in range(n):
                if i == 0:
                    action = action_list[int(options[i])][i, :]
                else:
                    action = np.vstack((action, action_list[int(options[i])][i, :]))

        return action
