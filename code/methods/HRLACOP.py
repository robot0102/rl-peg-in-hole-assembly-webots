import numpy as np
import torch
import torch.nn as nn
import random
import glob
from torch.autograd import Variable
import torch.nn.functional as F
from torch.distributions import Categorical
from ..utils.model import ActorList, Critic, OptionValue

if torch.cuda.is_available():
	torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LongTensor = torch.cuda.LongTensor


class HRLACOP(object):
	def __init__(self, args, state_dim, action_dim, max_action, option_num=2,
				 entropy_coeff=0.1, c_reg=1.0, c_ent=4, option_buffer_size=5000,
				 action_noise=0.2, policy_noise=0.2, noise_clip=0.5, use_option_net=True):
		
		self.args = args
		
		self.actor = ActorList(state_dim, action_dim, max_action, option_num).to(device)
		self.actor_target = ActorList(state_dim, action_dim, max_action, option_num).to(device)
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.args.learning_rate)
		
		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = Critic(state_dim, action_dim).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.args.learning_rate)
		
		if use_option_net:
			self.option = OptionValue(state_dim, option_num).to(device)
			self.option_target = OptionValue(state_dim, option_num).to(device)
			self.option_target.load_state_dict(self.option.state_dict())
			self.option_optimizer = torch.optim.Adam(self.option.parameters())
		
		self.use_option_net = use_option_net
		self.auxiliary_reward = self.args.auxiliary_reward
		self.max_action = max_action
		self.it = 0
		
		self.entropy_coeff = entropy_coeff
		self.c_reg = c_reg
		self.c_ent = c_ent
		
		self.option_buffer_size = option_buffer_size
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.option_num = option_num
		self.k = self.args.option_change
		self.action_noise = action_noise
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.q_predict = np.zeros(self.option_num)
		self.option_val = 0
	
	def train(self,
			  replay_buffer_lower,
			  replay_buffer_higher,
			  batch_size_lower=100,
			  batch_size_higher=100,
			  discount_higher=0.99,
			  discount_lower=0.99,
			  tau=0.005,
			  policy_freq=2):
		self.it += 1
		state, action, option, target_q = \
			self.calc_target_q(replay_buffer_lower, batch_size_lower, discount_lower, is_on_poliy=False)
		self.train_critic(state, action, target_q)
		
		# Delayed policy updates
		if self.it % policy_freq == 0:
			# option from option network
			# high_q_value, option_estimated = self.option(state)
			# max_option_idx = torch.argmax(option_estimated, dim=1)
			max_option_idx = self.select_option(state)
			action = self.actor(state)[torch.arange(state.shape[0]), :, max_option_idx]
			
			# ================ Train the actor =============================================#
			# off-policy learning :: sample state and option pair from replay buffer
			current_q_value, _ = self.critic(state, action)
			self.train_actor(current_q_value)
			# ===============================================================================#
			
			# update the frozen target networks
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
			
			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
		
		# Delayed option updates ::: on-policy
		if self.use_option_net:
			for _ in range(self.option_num):
				if len(replay_buffer_higher.storage) > batch_size_higher:
					state, option, target_q = \
						self.calc_target_option_q(replay_buffer_higher, batch_size_higher, discount_higher,
												  is_on_poliy=True)
					self.train_option(state, option, target_q)
					for param, target_param in zip(self.option.parameters(), self.option_target.parameters()):
						target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
	
	def train_critic(self, state, action, target_q):
		'''
		Calculate the loss of the critic and train the critic

		'''
		current_q1, current_q2 = self.critic(state, action)
		critic_loss = F.mse_loss(current_q1, target_q) + \
					  F.mse_loss(current_q2, target_q)
		
		# Three steps of training net using PyTorch:
		self.critic_optimizer.zero_grad()  # 1. Clear cumulative gradient
		critic_loss.backward()  # 2. Back propagation
		self.critic_optimizer.step()  # 3. Update the parameters of the net
	
	def train_actor(self, current_q_value):
		'''
		Calculate the loss of the actor and train the actor
		'''
		actor_loss = - current_q_value.mean()
		
		# Three steps of training net using PyTorch:
		self.actor_optimizer.zero_grad()  # 1. Clear cumulative gradient
		actor_loss.backward()  # 2. Back propagation
		self.actor_optimizer.step()  # 3. Update the parameters of the net
	
	def train_option(self, state, option, target_q):
		'''
		Calculate the loss of the option and train the option ：：：DQN
		'''
		current_q, _ = self.option(state)
		# print('current_q', current_q)
		# current_q = current_q.gather(1, option)
		# print('option', option)
		# print('current_q', current_q)
		current_q = current_q[:, option]
		option_loss = F.mse_loss(current_q, target_q)
		
		self.option_optimizer.zero_grad()  # 1. Clear cumulative gradient
		option_loss.backward()  # 2. Back propagation
		self.option_optimizer.step()  # 3. Update the parameters of the net
	
	def calc_target_q(self, replay_buffer, batch_size=100, discount=0.99, is_on_poliy=False):
		'''
		calculate q value for low-level policies
		'''
		# policy_noise = self.policy_noise
		# noise_clip = self.noise_clip
		if is_on_poliy:
			x, y, u, o, o_1, r, a_r, d = \
				replay_buffer.sample_on_policy(batch_size, self.option_buffer_size)
		else:
			x, y, u, o, o_1, r, a_r, d = \
				replay_buffer.sample(batch_size)
		
		state = torch.FloatTensor(x).to(device)
		if self.auxiliary_reward == True:
			_, average_r = self.calc_advantage_value(state, o)
			# print('auxiliary_reward', average_r)
			# print('reward', r)
			r = r + average_r.cpu().data.numpy()
		
		action = torch.FloatTensor(u).to(device)
		option = torch.FloatTensor(o).to(device)
		next_state = torch.FloatTensor(y).to(device)
		done = torch.FloatTensor(1 - d).to(device)
		reward = torch.FloatTensor(r).to(device)
		
		# need to be revised
		# next_option = torch.FloatTensor(o_1).to(device)
		# from gating policy
		next_option, _, q_predict = self.softmax_option_target(next_state)
		
		# inject noise and select next action
		noise = torch.FloatTensor(u).data.normal_(0, self.policy_noise).to(device)
		noise = noise.clamp(-self.noise_clip, self.noise_clip)
		next_action = (self.actor_target(next_state)[torch.arange(next_state.shape[0]), :, next_option]
					   + noise).clamp(-self.max_action, self.max_action)
		
		# for updating q-value
		target_q1, target_q2 = self.critic_target(next_state, next_action)
		target_q = torch.min(target_q1, target_q2)
		target_q = reward + (done * discount * target_q)
		
		return state, action, option, target_q
	
	def calc_target_option_q(self, replay_buffer, batch_size=100, discount=0.99, is_on_poliy=False):
		'''
		calculate option value
		'''
		if is_on_poliy:
			x, y, o, u, r = \
				replay_buffer.sample_on_policy(batch_size, self.option_buffer_size)
		else:
			x, y, o, u, r = \
				replay_buffer.sample(batch_size)
		
		state = torch.FloatTensor(x).to(device)
		next_state = torch.FloatTensor(y).to(device)
		option = torch.FloatTensor(o).to(device).long()
		
		# print('option', option)
		# next_option = torch.FloatTensor(u).to(device)
		reward = torch.FloatTensor(r).to(device)
		
		high_q_value, option_estimated = self.option_target(next_state)
		# high_q_value, option_estimated = self.option(next_state)
		max_option_idx = torch.argmax(option_estimated, dim=1)
		
		# print('training_q_value', high_q_value)
		# print('max_option_idx', max_option_idx)
		# next_q = torch.gather(high_q_value, 1, max_option_idx)
		next_q = high_q_value[:, max_option_idx]
		
		# next_q = high_q_value.gather(1, max_option_idx)
		target_q = reward + discount * next_q
		return state, option, target_q
	
	def calc_advantage_value(self, state, option):
		'''
		calculate advantage value
		'''
		option_value, _ = self.option_target(state)
		# option_value, _ = self.option(state)
		advantage_value = option_value - torch.mean(option_value)
		
		option = torch.LongTensor(option).to(device)
		advantage_value = torch.gather(advantage_value, 1, option)
		# advantage_value = advantage_value[torch.arange(state.shape[0]), :, option]
		
		# calculate the low-level auxiliary reward
		low_level_reward = advantage_value / self.args.option_change
		return advantage_value, low_level_reward
	
	def select_option(self, state):
		'''
		select options for training policies
		'''
		# state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		high_q_value, option_estimated = self.option(state)
		max_option_idx = torch.argmax(option_estimated, dim=1)
		return max_option_idx
	
	def select_action(self, state, option, change_option=False):
		'''
		for collection data
		'''
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		if change_option:
			option, _, q_predict = self.softmax_option_target(state)
			option = option.cpu().data.numpy().flatten()[0]
		# high_q_value, option_estimated = self.option(state)
		# option = torch.argmax(option_estimated, dim=1)
		action = self.actor(state)[torch.arange(state.shape[0]), :, option]
		return action.cpu().data.numpy().flatten(), option
	
	def select_evaluate_action(self, states):
		'''
		for evaluation
		'''
		states = torch.FloatTensor(states).to(device)
		option, _, q_predict = self.softmax_option_target(states)
		# evaluate obtain the argmax
		# option = option.cpu().data.numpy().flatten()[0]
		option = torch.argmax(q_predict, dim=1)
		# print('option', option)
		action = self.actor(states)[torch.arange(states.shape[0]), :, option]
		return action.cpu().data.numpy().flatten()
	
	def softmax_option_target(self, states):
		'''
		select new option every N or 2N steps
		'''
		# states = torch.FloatTensor(states).to(device)
		# q_predict = torch.zeros(states.shape[0], self.option_num, device=device)
		# for o in range(int(self.option_num)):
		# 	action_o = self.actor(states)[...,o]
		# 	q1, _ = self.critic_target(states, action_o)  # (batch_num, 1)
		# 	q_predict[:, o] = q1.squeeze()
		# Q_predict_i: B*O， B: batch number, O: option number
		batch_size = states.shape[0]
		action = self.actor(states)  # (batch_num, action_dim, option_num)
		option_num = action.shape[-1]
		# action: (batch_num, action_dim, option_num)-> (batch_num, option_num, action_dim)
		# -> (batch_num * option_num, action_dim)
		action = action.transpose(1, 2)
		action = action.reshape((-1, action.shape[-1]))
		# states: (batch_num, state_dim) -> (batch_num, state_dim * option_num)
		# -> (batch_num * option_num, state_dim)
		states = states.repeat(1, option_num).view(batch_size * option_num, -1)
		q_predict_1, _ = self.critic_target(states, action)
		# q_predict: (batch_num * option_num, 1) -> (batch_num, option_num)
		q_predict = q_predict_1.view(batch_size, -1)
		
		p = softmax(q_predict)
		o_softmax = p_sample(p)
		q_softmax = q_predict[:, o_softmax]
		return o_softmax, q_softmax, q_predict
	
	def save(self, filename, directory):
		torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
		torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
		torch.save(self.option.state_dict(), '%s/%s_option.pth' % (directory, filename))
	
	def load(self, filename, directory):
		actor_path = glob.glob('%s/%s_actor.pth' % (directory, filename))[0]
		self.actor.load_state_dict(torch.load(actor_path))
		critic_path = glob.glob('%s/%s_critic.pth' % (directory, filename))[0]
		print('actor path: {}, critic path: {}'.format(actor_path, critic_path))
		self.critic.load_state_dict(torch.load(critic_path))
		option_path = glob.glob('%s/%s_option.pth' % (directory, filename))[0]
		self.option.load_state_dict(torch.load(option_path))


def softmax(x):
	# This function is different from the Eq. 17, but it does not matter because
	# both the nominator and denominator are divided by the same value.
	# Equation 17: pi(o|s) = ext(Q^pi - max(Q^pi))/sum(ext(Q^pi - max(Q^pi))
	x_max, _ = torch.max(x, dim=1, keepdim=True)
	e_x = torch.exp(x - x_max)
	e_x_sum = torch.sum(e_x, dim=1, keepdim=True)
	out = e_x / e_x_sum
	return out


def p_sample(p):
	p_sum = torch.sum(p, dim=1, keepdim=True)
	p_normalized = p / p_sum
	m = Categorical(p_normalized)
	return m.sample()