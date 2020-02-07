import tensorflow as tf
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, merge, Lambda, Activation
from keras.layers.merge import Add, Multiply, Concatenate, concatenate
from keras.initializers import RandomUniform
from keras.optimizers import Adam
import keras.backend as K
from keras import metrics

import numpy.matlib

initializer = "glorot_uniform"  # Weight initilizer
final_initializer = RandomUniform(minval=-0.003, maxval=0.003)  # Weight initializer for the final layer

def weighted_entropy(p, w_norm):
    # w = tf.divide(tf.exp(A - np.max(A)), prob)
    # w_norm = w / K.sum(w)
    return K.sum( w_norm * p * K.log(p + 1e-8))

def weighted_mean(p, w_norm):
    # w = tf.exp(A- np.max(A))
    # w_norm = w / K.sum(w)
    p_weighted = tf.multiply(w_norm, p)
    return K.mean(p_weighted , axis=0 )

def weighted_mse( Q_target, Q_pred, w_norm ):
    # w = tf.exp(A- np.max(A))
    # w_norm = w / K.sum(w)
    error = K.square( Q_target - Q_pred )
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
    p_normalized = p  / np.matlib.repmat(p_sum, 1, col)
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
    x_out = x_input + at_eps * tf.multiply(epsilon, np.absolute(x_input))
    return x_out

def kl(p, q):
    return K.sum(p * K.log((p + 1e-8) / (q + 1e-8)))

class adInfoHRLTD3(object):
    def __init__(self, sess, env, state_dim, action_dim, action_bound, batch_size=64, tau=0.001, option_num=5,
                 actor_lr=0.0001, critic_lr=0.001, option_lr=0.001, gamma=0.99, hidden_dim=(400, 300),
                 entropy_coeff=0.1, c_reg=1.0, vat_noise=0.005, c_ent=4):
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
        self.option_num = option_num
        self.entropy_coeff = entropy_coeff
        self.c_reg = c_reg
        self.vat_noise = vat_noise
        self.c_ent = c_ent
        self.option_lr = option_lr

        # ===================================================================== #
        #                               Actor Model                             #
        # ===================================================================== #
        self.actor_state_input_list =[]
        self.actor_out_list =[]
        self.actor_model_list =[]
        self.actor_weight_list =[]
        for i in range(self.option_num):
            actor_state_input, actor_out, actor_model, actor_weights = self.create_actor_model()
            self.actor_state_input_list.append(actor_state_input)
            self.actor_out_list.append(actor_out)
            self.actor_model_list.append(actor_model)
            self.actor_weight_list.append(actor_weights)

        self.actor_target_state_input_list =[]
        self.actor_target_out_list =[]
        self.actor_target_model_list =[]
        self.actor_target_weight_list =[]
        for i in range(self.option_num):
            actor_target_state_input, actor_target_out, \
                    actor_target_model, actor_target_weights = self.create_actor_model()
            self.actor_target_state_input_list.append(actor_target_state_input)
            self.actor_target_out_list.append(actor_target_out)
            self.actor_target_model_list.append(actor_target_model)
            self.actor_target_weight_list.append(actor_target_weights)
        self.action_gradient_list = []
        for i in range(self.option_num):
            action_gradient = tf.placeholder(tf.float32, [None, self.action_dim])
            self.action_gradient_list.append(action_gradient)

        self.actor_optimizer_list = []
        for i in range(self.option_num):
            actor_params_grad = tf.gradients(self.actor_model_list[i].output, self.actor_weight_list[i], - self.action_gradient_list[i])
            grads = zip(actor_params_grad, self.actor_weight_list[i])
            self.actor_optimizer_list.append(tf.train.AdamOptimizer(self.actor_lr).apply_gradients(grads))

        # ===================================================================== #
        #                              Critic Model                             #
        # ===================================================================== #
        self.critic_state_input, self.critic_action_input, \
                    self.critic_out_Q1, self.critic_out_Q2, self.critic_model = self.create_critic_model()
        self.critic_target_state_input, self.critic_target_action_input, \
                    self.critic_out_Q1_target, self.critic_out_Q2_target, self.target_critic_model = self.create_critic_model()

        self.target_q_value = tf.placeholder(tf.float32, [None, 1])
        self.predicted_v_value = tf.placeholder(tf.float32, [None, 1])
        self.sampling_prob = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.critic_loss = metrics.mean_squared_error(self.target_q_value, self.critic_out_Q1) \
                           + metrics.mean_squared_error(self.target_q_value, self.critic_out_Q2)

        self.critic_optimize = tf.train.AdamOptimizer(self.critic_lr).minimize(self.critic_loss)

        # Get the gradient of the net w.r.t. the action.
        self.action_grads = tf.gradients(self.critic_out_Q1, self.critic_action_input)

        # ===================================================================== #
        #                              Option Model                             #
        # ===================================================================== #
        self.option_state_input, self.option_action_input, self.option_input_concat, self.option_out_dec, \
                                self.option_out, self.option_out_noise, self.option_model = self.create_option_model()
        Advantage = tf.stop_gradient(self.target_q_value - self.predicted_v_value)
        Weight = tf.divide(tf.exp(Advantage - np.max(Advantage)), self.sampling_prob)
        W_norm = Weight/K.mean(Weight)

        # H(o|s, a)
        critic_conditional_entropy = weighted_entropy(self.option_out, tf.stop_gradient(W_norm))
        p_weighted_ave = weighted_mean(self.option_out, tf.stop_gradient(W_norm))
        self.critic_entropy = critic_conditional_entropy - self.c_ent * entropy(p_weighted_ave)

        self.vat_loss = kl(self.option_out, self.option_out_noise)
        self.reg_loss = metrics.mean_absolute_error(self.option_input_concat, self.option_out_dec)
        self.option_loss = self.reg_loss + self.entropy_coeff * (self.critic_entropy) + self.c_reg * self.vat_loss
        self.option_optimize = tf.train.AdamOptimizer(self.option_lr).minimize(self.option_loss)

        # Initialize for later gradient calculations
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

    # ========================================================================= #
    #                               Model Architecture                          #
    # ========================================================================= #
    def create_actor_model(self):
        state_input = Input(shape=(self.state_dim,))

        h1 = Dense(self.hidden_dim[0], activation='relu',  kernel_initializer=initializer)(state_input)
        h2 = Dense(self.hidden_dim[1], activation='relu',  kernel_initializer=initializer)(h1)
        output = Dense(self.env.action_space.shape[0], activation='tanh', kernel_initializer=final_initializer)(h2)
        scaled_out = Lambda(self._scale2bound)(output)

        model = Model(input=state_input , output=[scaled_out])
        adam = Adam(lr=self.actor_lr)
        model.compile(loss="mse", optimizer=adam)
        return state_input, scaled_out, model, model.trainable_weights

    def create_critic_model(self):
        state_input = Input(shape=(self.state_dim,) )
        action_input = Input(shape=(self.action_dim, ))
        input = concatenate([state_input, action_input])

        h1_Q1 = Dense(self.hidden_dim[0], activation='relu',  kernel_initializer=initializer)(input)
        h2_Q1 = Dense(self.hidden_dim[1], activation='relu',  kernel_initializer=initializer)(h1_Q1)
        output_Q1 = Dense(1, activation='linear', kernel_initializer=final_initializer)(h2_Q1)

        h1_Q2 = Dense(self.hidden_dim[0], activation='relu', kernel_initializer=initializer)(input)
        h2_Q2 = Dense(self.hidden_dim[1], activation='relu', kernel_initializer=initializer)(h1_Q2)
        output_Q2 = Dense(1, activation='linear', kernel_initializer=final_initializer)(h2_Q2)

        model = Model(input=[state_input, action_input], output=[output_Q1, output_Q2])

        adam = Adam(lr=self.critic_lr)
        model.compile(loss="mse", optimizer=adam)
        return state_input, action_input, output_Q1, output_Q2, model

    def create_option_model(self):
        state_input = Input(shape=(self.state_dim,))
        action_input = Input(shape=(self.action_dim, ))
        input_concat = concatenate([state_input, action_input])

        Enc_1 = Dense(self.hidden_dim[0], activation='relu', kernel_initializer=initializer)
        Enc_2 = Dense(self.hidden_dim[1], activation='relu', kernel_initializer=initializer)
        Enc_3 = Dense(self.option_num, activation='linear', kernel_initializer=final_initializer)

        h1_option = Enc_1(input_concat)
        h2_option = Enc_2(h1_option)
        out_enc = Enc_3(h2_option)

        output_option = Activation('softmax')(out_enc)
        x2 = Lambda(self._add_normal)(input_concat)
        input_noise = Enc_1(x2)
        hidden2_noise = Enc_2(input_noise)
        out_noise = Enc_3(hidden2_noise)

        output_option_noise = Activation('softmax')(out_noise)

        # Generator
        Dec_1 = Dense(self.hidden_dim[1], activation='relu', kernel_initializer=initializer)
        Dec_2 = Dense(self.hidden_dim[0], activation='relu', kernel_initializer=initializer)
        Dec_3 = Dense(self.state_dim + self.action_dim, activation='linear', kernel_initializer=final_initializer)

        dec_1 = Dec_1(out_enc)
        dec_2 = Dec_2(dec_1)
        output_dec = Dec_3(dec_2)

        model = Model(input=[state_input, action_input], output=output_dec)

        adam = Adam(lr=self.critic_lr)
        model.compile(loss="mse", optimizer=adam)
        return state_input, action_input, input_concat, output_dec, output_option, output_option_noise, model

    # ========================================================================= #
    #                               Model Training                              #
    # ========================================================================= #
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

    def update_actor_target_network(self):
        for o in range(self.option_num):
            self.update_actor_target_network_option(o)

    def update_actor_target_network_option(self, option):
        actor_weights = self.actor_model_list[option].get_weights()
        actor_target_weights = self.actor_target_model_list[option].get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.tau * actor_weights[i] + (1 - self.tau) * actor_target_weights[i]

        self.actor_target_model_list[option].set_weights(actor_target_weights)

    def update_critic_target_network(self):
        critic_weights = self.critic_model.get_weights()
        critic_target_weights = self.target_critic_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.tau * critic_weights[i] + (1 - self.tau) * critic_target_weights[i]

        self.target_critic_model.set_weights(critic_target_weights)

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
        else:
            for i in range(n):
                if i == 0:
                    action = action_list[int(options[i])][i, :]
                else:
                    action = np.vstack((action, action_list[int(options[i])][i, :]))

        # print('action', action)
        return action

    def predict_actor_option_target(self, inputs, option):
        return self.sess.run(self.actor_target_out_list[option], feed_dict={self.actor_target_state_input_list[option]: inputs})

    def predict_actor_target(self, inputs, options):
        action_list = []
        for o in range(self.option_num):
            action_o = self.predict_actor_option_target(inputs, o)
            # print('action_o', action_o)
            action_list.append(action_o)

        # print('action_list', action_list)
        # print('options', options)
        n = inputs.shape[0]
        action = 0
        if n == 1 or np.isscalar(options):
            action = action_list[options]
        else:
            for i in range(n):
                if i == 0:
                    action = action_list[int(options[i])][i, :]
                else:
                    action = np.vstack((action, action_list[int(options[i])][i, :]))

        return action

    def predict_critic(self, inputs, actions):
        return self.sess.run([self.critic_out_Q1, self.critic_out_Q2], feed_dict={
            self.critic_state_input: inputs,
            self.critic_action_input: actions
        })

    def predict_critic_target(self, inputs, actions):
        return self.sess.run([self.critic_out_Q1_target, self.critic_out_Q2_target], feed_dict={
            self.critic_target_state_input: inputs,
            self.critic_target_action_input: actions
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.critic_state_input: inputs,
            self.critic_action_input: actions
        })

    def predict_option(self, inputs, actions):
        return self.sess.run([self.option_out], feed_dict={
            self.option_state_input: inputs,
            self.option_action_input: actions
        })

    def value_func(self, inputs):
        Q_predict = []
        n = inputs.shape[0]
        for o in range(int(self.option_num)):
            action_i = self.predict_actor_option_target(inputs, o)
            Q_predict_1, Q_predict_2= self.predict_critic_target(inputs, action_i)
            Q_predict_i = np.minimum(Q_predict_1, Q_predict_2)
            if o == 0:
                Q_predict = np.reshape( Q_predict_i, (-1, 1) )
            else:
                Q_predict = np.concatenate((Q_predict, np.reshape(Q_predict_i, (-1, 1)) ), axis= 1)

        po = softmax(Q_predict)
        state_values = weighted_mean_array(Q_predict, po)
        return state_values

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
        Q_softmax = Q_predict[np.arange(n) , o_softmax.flatten()]

        return o_softmax, np.reshape(Q_softmax, (n, 1)), Q_predict

    def max_option(self, inputs):
        Q_predict = []
        n = inputs.shape[0]
        for o in range(int(self.option_num)):
            action_i = self.predict_actor_target(inputs, o)
            Q_predict_i, _ = self.predict_critic_target(inputs, action_i)
            if o ==0:
                Q_predict = np.reshape( Q_predict_i, (-1, 1) )
            else:
                Q_predict = np.concatenate((Q_predict, np.reshape(Q_predict_i, (-1, 1)) ), axis= 1)

        o_max = np.argmax(Q_predict, axis=1)
        Q_max = np.max(Q_predict, axis=1)
        return o_max, Q_max, Q_predict

    def _scale2bound(self, inputs):
        scaled_out = tf.multiply(inputs, self.action_bound)
        return scaled_out

    def _add_normal(self, args):
        """
        Sampling from a normal distribution.
        """
        xx = args
        return add_normal(xx, tf.shape(xx), self.vat_noise)

    def save_model(self, iteration=-1, expname="unknown", model_path="./Model/"):
        """ Save models of policy """
        for o in range(self.option_num):
            self.actor_model_list[o].save(model_path + "%s_actor_option%d_iter%d.h5" % (expname, o, iteration))

        self.critic_model.save(model_path + "%s_critic_iter%d.h5" % (expname, iteration))
        self.option_model.save(model_path + "%s_option_mode_iter%d.h5" % (expname, iteration))

    def load_model(self, iteration=-1, expname="unknown", model_path="./Model/separate/"):
        for o in range(self.option_num):
            self.actor_model_list[o].load_weights(model_path + "%s_actor_option%d_iter%d.h5" % (expname, o, iteration))

        self.critic_model.load_weights(model_path + "%s_critic_iter%d.h5" % (expname, iteration))
        self.option_model.load_weights(model_path + "%s_option_mode_iter%d.h5" % (expname, iteration))
        print(model_path + "%s_actor_I%d.h5" % (expname, iteration))

