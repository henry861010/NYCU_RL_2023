import torch
import torch.nn as nn
import numpy as np
import os
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from replay_buffer.gae_replay_buffer import GaeSampleMemory
from base_agent import PPOBaseAgent
from models.atari_model import AtariNet
import gym
from environment_wrapper.racecarEnv import RaceCarEnvironment

class AtariPPOAgent(PPOBaseAgent):
	def __init__(self, config, actions):
		super(AtariPPOAgent, self).__init__(config)
		
		self.lr = config["learning_rate"]
		self.update_count = config["update_ppo_epoch"]
		self.actions = actions
		self.frames = config["frames_stack"]

		self.env = RaceCarEnvironment(N_frame=self.frames, test=False, scenario="austria_competition_collisionStop") 
		self.test_env = RaceCarEnvironment(N_frame=self.frames, test=True, scenario="austria_competition")

		self.net = AtariNet(len(actions), self.env.observation_space.shape[1], self.frames, True) 
		self.net.to(self.device)
		self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)
		
	def decide_agent_actions(self, observation, eval=False):
		observation = torch.tensor(np.array([observation]), dtype=torch.float).to(self.device)
		if eval:
			with torch.no_grad():
				_, act_dist, value, entropy = self.net(observation)
				action = act_dist.probs.argmax().view(1)
		else:
			action, act_dist, value, entropy = self.net(observation)
		action_env = self.actions[action.item()]
		return np.array(action_env), action.cpu() ,act_dist.probs.gather(1, action.unsqueeze(1)).cpu().item(), value.cpu().item(), entropy.cpu().item()

	
	def update(self):
		loss_counter = 0.0001
		total_surrogate_loss = 0
		total_v_loss = 0
		total_entropy = 0
		total_loss = 0

		batches = self.gae_replay_buffer.extract_batch(self.discount_factor_gamma, self.discount_factor_lambda)
		sample_count = len(batches["action"])
		batch_index = np.random.permutation(sample_count)
		
		observation_batch = {}
		for key in batches["observation"]:
			observation_batch[key] = batches["observation"][key][batch_index]
		action_batch = batches["action"][batch_index]
		return_batch = batches["return"][batch_index]
		adv_batch = batches["adv"][batch_index]
		v_batch = batches["value"][batch_index]
		logp_pi_batch = batches["logp_pi"][batch_index]

		for _ in range(self.update_count):
			for start in range(0, sample_count, self.batch_size):
				ob_train_batch = {}
				for key in observation_batch:
					ob_train_batch[key] = observation_batch[key][start:start + self.batch_size]
				ac_train_batch = action_batch[start:start + self.batch_size]
				return_train_batch = return_batch[start:start + self.batch_size]
				adv_train_batch = adv_batch[start:start + self.batch_size]
				v_train_batch = v_batch[start:start + self.batch_size]
				logp_pi_train_batch = logp_pi_batch[start:start + self.batch_size]

				ob_train_batch = torch.from_numpy(ob_train_batch["observation_2d"])
				ob_train_batch = ob_train_batch.to(self.device, dtype=torch.float32)

				ac_train_batch = torch.from_numpy(ac_train_batch)
				ac_train_batch = ac_train_batch.to(self.device, dtype=torch.long)

				adv_train_batch = torch.from_numpy(adv_train_batch)
				adv_train_batch = adv_train_batch.to(self.device, dtype=torch.float32)

				logp_pi_train_batch = torch.from_numpy(logp_pi_train_batch)
				logp_pi_train_batch = logp_pi_train_batch.to(self.device, dtype=torch.float32)

				return_train_batch = torch.from_numpy(return_train_batch)
				return_train_batch = return_train_batch.to(self.device, dtype=torch.float32)


				### TODO ###
				# calculate loss and update network
				_, act_dist, current_value, entropy = self.net(ob_train_batch)

				#logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
				current_logp = act_dist.probs.gather(1, ac_train_batch).view(-1)
				entropy = torch.mean(entropy)

				# calculate policy loss
				ratio = current_logp.to(self.device)/(logp_pi_train_batch.detach()+0.0000000000001)
				#ratios = torch.exp(logprobs - old_logprobs.detach())

				surr1 = ratio * adv_train_batch
				surr2 = torch.clamp(ratio, 1 - self.clip_epsilon,1 + self.clip_epsilon) * adv_train_batch
				surrogate_loss = torch.mean(-torch.min(surr1, surr2))
				#surr1 = ratios * advantages
            	#surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

				# calculate value loss
				value_criterion = nn.MSELoss()
				v_loss = value_criterion( current_value, return_train_batch.detach() )
				
				# calculate total loss
				loss = surrogate_loss + self.value_coefficient * v_loss - self.entropy_coefficient * entropy
				## loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

				# update network
				self.optim.zero_grad()
				loss.backward()
				nn.utils.clip_grad_norm_(self.net.parameters(), self.max_gradient_norm)
				self.optim.step()

				total_surrogate_loss += surrogate_loss.item()
				total_v_loss += v_loss.item()
				total_entropy += entropy.item()
				total_loss += loss.item()
				loss_counter += 1

		self.writer.add_scalar('PPO/Loss', total_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Surrogate Loss', total_surrogate_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Value Loss', total_v_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Entropy', total_entropy / loss_counter, self.total_time_step)
		print(f"Loss: {total_loss / loss_counter}\
			\tSurrogate Loss: {total_surrogate_loss / loss_counter}\
			\tValue Loss: {total_v_loss / loss_counter}\
			\tEntropy: {total_entropy / loss_counter}\
			")
	



