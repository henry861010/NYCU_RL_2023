import torch
import torch.nn as nn
import numpy as np
from base_agent import BaseAgent
from models.CarRacing_model import ActorNetSimple, CriticNetSimple
from environment_wrapper.racecarEnv import RaceCarEnvironment
import random
import gym
import torch.nn.functional as F

class SACAgent(BaseAgent):
	def __init__(self, config, actions):
		super(SACAgent, self).__init__(config)
		# initialize environment
		self.env = RaceCarEnvironment(N_frame=self.frames, test=False) 
		self.test_env = RaceCarEnvironment(N_frame=self.frames, test=True)

		self.actions = actions
		
		print("*******self.env.observation_space.shape[1]: ",self.env.observation_space.shape[1])
		print("*******self.env.action_space.shape[0]: ",self.env.action_space.shape[0])
		# behavior network
		self.actor_net = ActorNetSimple(self.env.observation_space.shape[1], len(self.actions), self.frames).to(self.device)
		self.critic_net1 = CriticNetSimple(self.env.observation_space.shape[1], len(self.actions), self.frames).to(self.device)
		self.critic_net2 = CriticNetSimple(self.env.observation_space.shape[1], len(self.actions), self.frames).to(self.device)
		
		# target network
		self.target_critic_net1 = CriticNetSimple(self.env.observation_space.shape[1], len(self.actions), self.frames).to(self.device)
		self.target_critic_net2 = CriticNetSimple(self.env.observation_space.shape[1], len(self.actions), self.frames).to(self.device)
		self.target_critic_net1.load_state_dict(self.critic_net1.state_dict())
		self.target_critic_net2.load_state_dict(self.critic_net2.state_dict())
		
		# set optimizer
		self.lra = config["lra"]
		self.lrc = config["lrc"]
		self.temperature_lr = config["temperature_lr"] 
		
		self.log_temperature =  torch.tensor(np.log(0.01), dtype=torch.float)
		self.log_temperature.requires_grad = True
		# set target entropy to -|A|
		self.target_entropy = -self.env.action_space.shape[0]
		
		self.actor_opt = torch.optim.Adam(self.actor_net.parameters(), lr=self.lra)
		self.critic_opt1 = torch.optim.Adam(self.critic_net1.parameters(), lr=self.lrc)
		self.critic_opt2 = torch.optim.Adam(self.critic_net2.parameters(), lr=self.lrc)
		self.log_temperature_opt = torch.optim.Adam([self.log_temperature], lr=self.temperature_lr)

	
	def decide_agent_actions(self, state):
		state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
		probs = self.actor_net(state)
		action_dist = torch.distributions.Categorical(probs)
		action_num = action_dist.sample()

		action_env = self.actions[action_num.item()]

		return action_env,action_num.cpu().item()
		

	def update_behavior_network(self):
		# sample a minibatch of transitions
		state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)

		action = action.to(torch.int64)

		### update critic network ###
		next_probs = self.actor_net(next_state)
		next_log_probs = torch.log(next_probs + 1e-8)

		entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)

		q1_next = self.target_critic_net1(next_state)
		q2_next = self.target_critic_net2(next_state)
		
		next_value = torch.sum(next_probs * torch.min(q1_next, q2_next),dim=1,keepdim=True) + self.log_temperature.exp()*entropy
		q_target= (reward + ~done.bool() * self.gamma * next_value)

		
		criterion = nn.MSELoss()
		critic_loss1 = torch.mean(F.mse_loss(self.critic_net1(state).gather(1, action), q_target.detach()))
		critic_loss2 = torch.mean(F.mse_loss(self.critic_net2(state).gather(1, action), q_target.detach()))

		self.critic_opt1.zero_grad()
		critic_loss1.backward()
		self.critic_opt1.step()

		self.critic_opt2.zero_grad()
		critic_loss2.backward()
		self.critic_opt2.step()

		### update policy network ###
		prob_now = self.actor_net(state)
		log_probs = torch.log(prob_now + 1e-8)
		entropy = -torch.sum(prob_now * log_probs, dim=1, keepdim=True)
		q1 = self.critic_net1(state)
		q2 = self.critic_net2(state)
		
		min_qvalue = torch.sum(prob_now * torch.min(q1, q2),dim=1,keepdim=True)
		policy_loss = torch.mean(-self.log_temperature.exp() * entropy - min_qvalue)

		self.actor_opt.zero_grad()
		policy_loss.backward()
		self.actor_opt.step()

		### update temperature ###
		temperature_loss = (self.log_temperature.exp()*(entropy - self.target_entropy).detach()).mean()

		self.log_temperature_opt.zero_grad()
		temperature_loss.backward()
		self.log_temperature_opt.step()


		
