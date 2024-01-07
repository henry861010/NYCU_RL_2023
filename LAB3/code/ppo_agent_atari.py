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
from wrapper.resize_observation import ResizeObservation
from wrapper.gray_scale_observation import GrayScaleObservation
from wrapper.frame_stack import FrameStack
from wrapper.get_part_observation import GetPartObservation

class AtariPPOAgent(PPOBaseAgent):
	def __init__(self, config):
		super(AtariPPOAgent, self).__init__(config)
		### TODO ###
		# initialize env
		self.env = gym.make("ALE/Enduro-v5")
		self.env = GetPartObservation(self.env)
		self.env = ResizeObservation(self.env, 84) #https://gymnasium.farama.org/api/wrappers/observation_wrappers/#gymnasium.wrappers.ResizeObservation
		self.env = GrayScaleObservation(self.env)  #https://gymnasium.farama.org/api/wrappers/observation_wrappers/#gymnasium.wrappers.GrayScaleObservation
		self.env = FrameStack(self.env, 4)  #https://gymnasium.farama.org/api/wrappers/observation_wrappers/#gymnasium.wrappers.FrameStack 

	
		### TODO ###
		# initialize test_env
		self.test_env = gym.make("ALE/Enduro-v5", render_mode='human')
		self.test_env = GetPartObservation(self.test_env)
		self.test_env = ResizeObservation(self.test_env, 84)
		self.test_env = GrayScaleObservation(self.test_env)
		self.test_env = FrameStack(self.test_env, 4)

		self.net = AtariNet(self.env.action_space.n)
		self.net.to(self.device)
		self.lr = config["learning_rate"]
		self.update_count = config["update_ppo_epoch"]
		self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)
		
	def decide_agent_actions(self, observation, eval=False):
		observation = torch.tensor(np.array([observation]), dtype=torch.float).to(self.device)
		if eval:
			with torch.no_grad():
				_, act_dist, value, entropy = self.net(observation)
				action = act_dist.probs.argmax().view(1)
		else:
			action, act_dist, value, entropy = self.net(observation)
		return action.cpu().detach(), act_dist.probs.gather(1, action.unsqueeze(1)).cpu().item(), value.cpu().item(), entropy.cpu().item()

	
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


				# calculate loss and update network
				_, act_dist, current_value, entropy = self.net(ob_train_batch)

				#logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
				current_logp = act_dist.probs.gather(1, ac_train_batch).view(-1)
				entropy = torch.mean(entropy)

				# calculate policy loss
				ratio = current_logp.to(self.device)/(logp_pi_train_batch.detach()+0.0000000000001)

				surr1 = ratio * adv_train_batch
				surr2 = torch.clamp(ratio, 1 - self.clip_epsilon,1 + self.clip_epsilon) * adv_train_batch
				surrogate_loss = torch.mean(-torch.min(surr1, surr2))

				# calculate value loss
				value_criterion = nn.MSELoss()
				v_loss = value_criterion( current_value, return_train_batch.detach() )
				
				# calculate total loss
				loss = surrogate_loss + self.value_coefficient * v_loss - self.entropy_coefficient * entropy

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
	



