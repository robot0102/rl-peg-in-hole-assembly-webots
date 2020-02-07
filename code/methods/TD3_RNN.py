import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import glob
if torch.cuda.is_available():
	torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, hidden_layer=1,
				 l1_hidden_dim = 400, l2_hidden_dim = 300):
		super().__init__()
		self.gru = nn.GRU(state_dim, l1_hidden_dim, hidden_layer, batch_first=True)
		self.l2 = nn.Linear(l1_hidden_dim, l2_hidden_dim)
		self.l3 = nn.Linear(l2_hidden_dim, action_dim)
		self.max_action = max_action

	def forward(self, x):
		x, _ = self.gru(x)
		x = x[:, -1, :]
		x = F.relu(self.l2(x))
		x = self.max_action * torch.tanh(self.l3(x))
		return x


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, l1_hidden_dim = 300, hidden_layer=1):
		super(Critic, self).__init__()

		# Q1 architecture
		self.gru1 = nn.GRU(state_dim, l1_hidden_dim, hidden_layer, batch_first=True)
		self.l1 = nn.Linear(action_dim, 100)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, 1)

		# Q2 architecture
		self.gru2 = nn.GRU(state_dim, l1_hidden_dim, hidden_layer, batch_first=True)
		self.l4 = nn.Linear(action_dim, 100)
		self.l5 = nn.Linear(400, 300)
		self.l6 = nn.Linear(300, 1)


	def forward(self, x, u):
		xg1, _ = self.gru1(x)
		xg1 = xg1[:, -1, :]
		u1 = F.relu(self.l1(u))
		xu1 = torch.cat([xg1, u1], 1)

		x1 = F.relu(self.l2(xu1))
		x1 = self.l3(x1)

		xg2, _ = self.gru2(x)
		xg2 = xg2[:, -1, :]
		u2 = F.relu(self.l4(u))
		xu2 = torch.cat([xg2, u2], 1)

		x2 = F.relu(self.l5(xu2))
		x2 = self.l6(x2)

		return x1, x2


class TD3_RNN(object):
	def __init__(self, state_dim, action_dim, max_action):
		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = Critic(state_dim, action_dim).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

		self.max_action = max_action
		self.it = 0


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(-1, state.shape[0], state.shape[1])).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def cal_estimate_value(self, replay_buffer, eval_states=10000):
		x, _, u, _, _ = replay_buffer.sample(eval_states)
		state = torch.FloatTensor(x).to(device)
		action = torch.FloatTensor(u).to(device)
		Q1, Q2 = self.critic(state, action)
		# target_Q = torch.mean(torch.min(Q1, Q2))
		Q_val = 0.5 * (torch.mean(Q1) + torch.mean(Q2))
		return Q_val.detach().cpu().numpy()


	def train(self, replay_buffer, batch_size=100, discount=0.99, tau=0.005,
			  policy_noise=0.2, noise_clip=0.5, policy_freq=2):
		self.it += 1
		# Sample replay buffer
		x, y, u, r, d = replay_buffer.sample(batch_size)
		state = torch.FloatTensor(x).to(device)
		action = torch.FloatTensor(u).to(device)
		next_state = torch.FloatTensor(y).to(device)
		done = torch.FloatTensor(1 - d).to(device)
		reward = torch.FloatTensor(r).to(device)

		# Select action according to policy and add clipped noise
		noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(device)
		noise = noise.clamp(-noise_clip, noise_clip)
		next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

		# Compute the target Q value
		target_Q1, target_Q2 = self.critic_target(next_state, next_action)

		target_Q = torch.min(target_Q1, target_Q2)
		target_Q = reward + (done * discount * target_Q).detach()

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.it % policy_freq == 0:

			# Compute actor loss
			current_Q1, current_Q2 = self.critic(state, self.actor(state))
			actor_loss = - current_Q1.mean()

			# Optimize the actor
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


	def save(self, filename, directory):
		torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
		torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

	def load(self, filename, directory):
		actor_path = glob.glob('%s/%s_actor.pth' % (directory, filename))[0]
		self.actor.load_state_dict(torch.load(actor_path))
		critic_path = glob.glob('%s/%s_critic.pth' % (directory, filename))[0]
		print('actor path: {}, critic path: {}'.format(actor_path, critic_path))
		self.critic.load_state_dict(torch.load(critic_path))
