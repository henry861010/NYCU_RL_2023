import torch
import torch.nn as nn
import numpy as np
from base_agent import BaseAgent
from models.CarRacing_model import ActorNetSimple, CriticNetSimple
from environment_wrapper.racecarEnv import RaceCarEnvironment
import random
import gym

class SACAgent(BaseAgent):
	def __init__(self, config):
		super(SACAgent, self).__init__(config)
		# initialize environment
		self.env = RaceCarEnvironment(N_frame=self.frames, test=False)   ###############
		self.test_env = RaceCarEnvironment(N_frame=self.frames, test=True)

		print("*******self.env.observation_space.shape[1]: ",self.env.observation_space.shape[1])
		print("*******self.env.action_space.shape[0]: ",self.env.action_space.shape[0])
		# behavior network
		self.actor_net = ActorNetSimple(self.env.observation_space.shape[1], self.env.action_space.shape[0], self.frames, self.game, self.obImage).to(self.device)
		self.critic_net1 = CriticNetSimple(self.env.observation_space.shape[1], self.env.action_space.shape[0], self.frames, self.game, self.obImage).to(self.device)
		self.critic_net2 = CriticNetSimple(self.env.observation_space.shape[1], self.env.action_space.shape[0], self.frames, self.game, self.obImage).to(self.device)
		
		# target network
		self.target_critic_net1 = CriticNetSimple(self.env.observation_space.shape[1], self.env.action_space.shape[0], self.frames, self.game, self.obImage).to(self.device)
		self.target_critic_net2 = CriticNetSimple(self.env.observation_space.shape[1], self.env.action_space.shape[0], self.frames, self.game, self.obImage).to(self.device)
		self.target_critic_net1.load_state_dict(self.critic_net1.state_dict())
		self.target_critic_net2.load_state_dict(self.critic_net2.state_dict())
		
		# set optimizer
		self.lra = config["lra"]
		self.lrc = config["lrc"]
		self.temperature_lr = config["temperature_lr"] 
		
		self.init_temperature = config["init_temperature"]
		self.log_temperature = torch.tensor(np.log(self.init_temperature)).to(self.device)
		self.log_temperature.requires_grad = True
		# set target entropy to -|A|
		self.target_entropy = -self.env.action_space.shape[0]
		
		self.actor_opt = torch.optim.Adam(self.actor_net.parameters(), lr=self.lra)
		self.critic_opt1 = torch.optim.Adam(self.critic_net1.parameters(), lr=self.lrc)
		self.critic_opt2 = torch.optim.Adam(self.critic_net2.parameters(), lr=self.lrc)
		self.log_temperature_opt = torch.optim.Adam([self.log_temperature], lr=self.temperature_lr)

	
	def decide_agent_actions(self, state):
		
		state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
		action, _ = self.actor_net(state)
		action = action.detach().flatten().cpu().numpy()

		return action
		

	def update_behavior_network(self):
		# sample a minibatch of transitions
		state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)

		### update critic network ###
		next_action, log_prob_next = self.actor_net(next_state)
		q1_next = self.target_critic_net1(next_state, next_action)
		q2_next = self.target_critic_net2(next_state, next_action)
		
		next_value = torch.min(q1_next, q2_next) - self.log_temperature.exp()*log_prob_next
		q_target= (reward + ~done.bool() * self.gamma * next_value)

		criterion = nn.MSELoss()
		critic_loss1 = criterion(self.critic_net1(state, action), q_target.detach())
		critic_loss2 = criterion(self.critic_net2(state, action), q_target.detach())

		self.critic_opt1.zero_grad()
		critic_loss1.backward()
		self.critic_opt1.step()

		self.critic_opt2.zero_grad()
		critic_loss2.backward()
		self.critic_opt2.step()

		### update policy network ###
		now_action, log_prob_now = self.actor_net(state)
		q1 = self.target_critic_net1(state, now_action)
		q2 = self.target_critic_net2(state, now_action)
		
		policy_loss = (self.log_temperature.exp()*log_prob_now - torch.min(q1, q2)).mean()

		self.actor_opt.zero_grad()
		policy_loss.backward()
		self.actor_opt.step()

		### update temperature ###
		temperature_loss = (self.log_temperature.exp()*(-log_prob_now - self.target_entropy).detach()).mean()

		self.log_temperature_opt.zero_grad()
		temperature_loss.backward()
		self.log_temperature_opt.step()


		
