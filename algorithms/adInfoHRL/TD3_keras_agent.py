import tensorflow as tf

import numpy as np
import gym
from gym import wrappers
import mujoco_py

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, merge, Lambda, Activation
from keras.layers.merge import Add, Multiply, Concatenate, concatenate
from keras.optimizers import Adam
from keras.initializers import RandomUniform
import keras.backend as K
from keras import metrics

from replay_buffer import ReplayBuffer

import argparse
import pprint as pp

initializer = "glorot_uniform"  # Weight initilizer
final_initializer = RandomUniform(minval=-0.003, maxval=0.003)  # Weight initializer for the final layer

class TD3(object):
    def __init__(self, sess, env, state_dim, action_dim, action_bound, batch_size=64, tau=0.001,
                 actor_lr=0.0001, critic_lr=0.001, gamma = 0.99, hidden_dim=(400, 300)):
        self.env = env
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim

        # ===================================================================== #
        #                               Actor Model                             #
        # ===================================================================== #

        self.actor_state_input, self.actor_scaled_out, \
            self.actor_model, self.actor_weights = self.create_actor_model()


        self.actor_target_state_input, self.actor_target_scaled_out, \
            self.target_actor_model, self.actor_target_weights = self.create_actor_model()

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.action_dim])

        self.actor_params_grad = tf.gradients(self.actor_model.output, self.actor_weights, -self.action_gradient)
        grads = zip(self.actor_params_grad, self.actor_weights)

        self.actor_optimize = tf.train.AdamOptimizer(self.actor_lr).apply_gradients(grads)

        # ===================================================================== #
        #                              Critic Model                             #
        # ===================================================================== #

        self.critic_state_input, self.critic_action_input, \
            self.critic_out_Q1, self.critic_out_Q2, self.critic_model = self.create_critic_model()

        self.critic_target_state_input, self.critic_target_action_input, \
        self.critic_out_Q1_target, self.critic_out_Q2_target, self.target_critic_model = self.create_critic_model()

        # Network target (y_i)
        self.target_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.critic_loss = metrics.mean_squared_error(self.target_q_value, self.critic_out_Q1) + metrics.mean_squared_error(self.target_q_value, self.critic_out_Q2)

        self.critic_optimize = tf.train.AdamOptimizer( self.critic_lr).minimize(self.critic_loss)

        # Get the gradient of the net w.r.t. the action.
        self.action_grads = tf.gradients(self.critic_out_Q1, self.critic_action_input)

        # Initialize for later gradient calculations
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

    # ========================================================================= #
    #                               Model Architecture                          #
    # ========================================================================= #
    def create_actor_model(self):
        state_input = Input(shape=(self.state_dim,))
        h1 = Dense(self.hidden_dim[0], activation='relu', kernel_initializer=initializer)(state_input)
        h2 = Dense(self.hidden_dim[1], activation='relu', kernel_initializer=initializer)(h1)
        output = Dense(self.env.action_space.shape[0], activation='tanh', kernel_initializer=final_initializer)(h2)
        scaled_out = Lambda(self._scale2bound)(output)

        model = Model(input=state_input, output=scaled_out)
        adam = Adam(lr=self.actor_lr)
        model.compile(loss="mse", optimizer=adam)
        return state_input, scaled_out, model, model.trainable_weights

    def create_critic_model(self):
        state_input = Input(shape=(self.state_dim,) )
        action_input = Input(shape=(self.action_dim, ))
        input = concatenate([state_input, action_input])

        h1_Q1 = Dense(self.hidden_dim[0], activation='relu', kernel_initializer=initializer)(input)
        h2_Q1 = Dense(self.hidden_dim[1], activation='relu', kernel_initializer=initializer)(h1_Q1)
        output_Q1 = Dense(1, activation='linear', kernel_initializer=final_initializer)(h2_Q1)

        h1_Q2 = Dense(self.hidden_dim[0], activation='relu', kernel_initializer=initializer)(input)
        h2_Q2 = Dense(self.hidden_dim[1], activation='relu',kernel_initializer=initializer)(h1_Q2)
        output_Q2 = Dense(1, activation='linear', kernel_initializer=final_initializer)(h2_Q2)

        model = Model(input=[state_input, action_input], output=[output_Q1, output_Q2])

        adam = Adam(lr=self.critic_lr)
        model.compile(loss="mse", optimizer=adam)
        return state_input, action_input, output_Q1, output_Q2, model

    # ========================================================================= #
    #                               Model Training                              #
    # ========================================================================= #
    def train_actor(self, inputs, a_gradient):
        self.sess.run(self.actor_optimize, feed_dict={
            self.actor_state_input: inputs,
            self.action_gradient: a_gradient
        })

    def train_critic(self, inputs, action, predicted_q_value):
        return self.sess.run([self.critic_optimize], feed_dict={
            self.critic_state_input: inputs,
            self.critic_action_input: action,
            self.target_q_value: predicted_q_value
        })

    def update_actor_target_network(self):
        actor_weights = self.actor_model.get_weights()
        actor_target_weights = self.target_actor_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.tau * actor_weights[i] + (1 - self.tau) * actor_target_weights[i]

        self.target_actor_model.set_weights(actor_target_weights)

    def update_critic_target_network(self):
        critic_weights = self.critic_model.get_weights()
        critic_target_weights = self.target_critic_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.tau * critic_weights[i] + (1 - self.tau) * critic_target_weights[i]

        self.target_critic_model.set_weights(critic_target_weights)

    def predict_actor(self, inputs):
        return self.sess.run(self.actor_scaled_out, feed_dict={
            self.actor_state_input: inputs
        })

    def predict_critic_target(self, inputs, actions):
        return self.sess.run([self.critic_out_Q1_target, self.critic_out_Q2_target], feed_dict={
            self.critic_target_state_input: inputs,
            self.critic_target_action_input: actions
        })

    def predict_actor_target(self, inputs):
        return self.sess.run(self.actor_target_scaled_out, feed_dict={
            self.actor_target_state_input: inputs
        })

    def action_gradients(self,inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.critic_state_input: inputs,
            self.critic_action_input: actions
        })

    def _scale2bound(self, inputs):
        scaled_out = tf.multiply(inputs, self.action_bound)
        return scaled_out

    def save_model(self, iteration=-1, expname="unknown", model_path = "./Model/"):
        """ Save models of policy """
        self.actor_model.save(model_path + "%s_actor_I%d.h5" % (expname, iteration))
        self.target_actor_model.save(model_path + "%s_target_actor_iter%d.h5" % (expname, iteration))
        self.critic_model.save(model_path + "%s_critic_I%d.h5" % (expname, iteration))
        self.target_critic_model.save(model_path + "%s_target_critic_iter%d.h5" % (expname, iteration))

    def load_model(self, iteration=-1, expname="unknown", model_path = "./Model/"):
        self.actor_model.load_weights(model_path + "%s_actor_I%d.h5" % (expname, iteration))
        # self.target_actor_model.load_weights(model_path + "%s_target_actor_iter%d.h5" % (expname, iteration))
        self.critic_model.load_weights(model_path + "%s_critic_I%d.h5" % (expname, iteration))
        # self.target_critic_model.load_weights(model_path + "%s_target_critic_iter%d.h5" % (expname, iteration))
        print(model_path + "%s_actor_I%d.h5" % (expname, iteration))
