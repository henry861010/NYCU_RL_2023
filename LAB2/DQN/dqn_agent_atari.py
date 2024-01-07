import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from base_agent import DQNBaseAgent
from models.atari_model import AtariNetDQN
import gym
import random
from wrapper.resize_observation import ResizeObservation
from wrapper.gray_scale_observation import GrayScaleObservation
from wrapper.frame_stack import FrameStack
from wrapper.get_part_observation import GetPartObservation
import torch.nn.functional as F

class AtariDQNAgent(DQNBaseAgent):
	def __init__(self, config):
		super(AtariDQNAgent, self).__init__(config)

		# initialize env
		self.env = gym.make("ALE/MsPacman-v5")
		self.env = GetPartObservation(self.env)
		self.env = ResizeObservation(self.env, 84) 
		self.env = GrayScaleObservation(self.env) 
		self.env = FrameStack(self.env, 4)

		# initialize test_env
		self.test_env = gym.make("ALE/MsPacman-v5", render_mode='human')
		self.test_env = GetPartObservation(self.test_env)
		self.test_env = ResizeObservation(self.test_env, 84)
		self.test_env = GrayScaleObservation(self.test_env) 
		self.test_env = FrameStack(self.test_env, 4)

		# initialize behavior network and target network
		self.behavior_net = AtariNetDQN(self.env.action_space.n)
		self.behavior_net.to(self.device)
		self.target_net = AtariNetDQN(self.env.action_space.n)
		self.target_net.to(self.device)
		self.target_net.load_state_dict(self.behavior_net.state_dict())
		# initialize optimizer
		self.lr = config["learning_rate"]
		self.optim = torch.optim.Adam(self.behavior_net.parameters(), lr=self.lr, eps=1.5e-4)
		
	def decide_agent_actions(self, observation, epsilon=0.0, action_space=None):
		if random.random() < epsilon:
			action = np.random.randint(9)
		else:
			observation = torch.tensor(np.array([observation]), dtype=torch.float).to(self.device)
			with torch.no_grad():
				action = self.behavior_net(observation).argmax().item()

		return action

	def update_behavior_network(self):
		state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)

		action = action.to(torch.int64)
		q_value = self.behavior_net(state).gather(1,action)
		with torch.no_grad():
			q_next = self.target_net(next_state).max(1)[0].view(-1, 1)
			q_target = reward + self.gamma *(1 - done)*q_next
        
		
		loss = torch.mean(F.mse_loss(q_value, q_target))

		self.writer.add_scalar('DQN/Loss', loss.item(), self.total_time_step)

		self.optim.zero_grad()
		loss.backward()
		self.optim.step()

	
	