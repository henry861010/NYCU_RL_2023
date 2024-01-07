import torch
import torch.nn as nn
import numpy as np
import os
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from replay_buffer.replay_buffer import ReplayMemory
from abc import ABC, abstractmethod
import requests
import json
from collections import deque
import cv2
from datetime import datetime
from environment_wrapper.final_project_env.racecar_gym.env import RaceEnv
from pathlib import Path


class BaseAgent(ABC):
	def __init__(self, config):
		self.gpu = config["gpu"]
		self.device = torch.device("cuda" if self.gpu and torch.cuda.is_available() else "cpu")
		self.total_time_step = int(config["already_time_step"])
		self.now_retrain_step = 0
		self.training_steps = int(config["training_steps"])
		self.batch_size = int(config["batch_size"])
		self.warmup_steps = config["warmup_steps"]
		self.total_episode = config["total_episode"]
		self.eval_interval = config["eval_interval"]
		self.eval_episode = config["eval_episode"]
		self.gamma = config["gamma"]
		self.tau = config["tau"]
		self.update_freq = config["update_freq"]
		self.obImage =config["observatrion_Image"]
		self.frames = config["frames"]
	
		self.replay_buffer = ReplayMemory(int(config["replay_buffer_capacity"]))
		self.writer = SummaryWriter(config["logdir"])

		self.images = []

	@abstractmethod
	def decide_agent_actions(self, state):
		
		return NotImplementedError
		
	
	def update(self):
		# update the behavior networks
		self.update_behavior_network()
		# update the target networks
		if self.total_time_step % self.update_freq == 0:
			self.update_target_network(self.target_critic_net1, self.critic_net1, self.tau)
			self.update_target_network(self.target_critic_net2, self.critic_net2, self.tau)

	@abstractmethod
	def update_behavior_network(self):
		# sample a minibatch of transitions
		state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)

		return NotImplementedError
		

	@staticmethod
	def update_target_network(target_net, net, tau):
		# update target network by "soft" copying from behavior network
		for target, behavior in zip(target_net.parameters(), net.parameters()):
			target.data.copy_((1 - tau) * target.data + tau * behavior.data)

	def train(self):
		for episode in range(self.total_episode):
			total_reward = 0
			state, infos = self.env.reset()
			for t in range(1000000):
				if self.now_retrain_step < self.warmup_steps:
					action_num = 1
					action = self.actions[action_num]
				else:
					action, action_num = self.decide_agent_actions(state)
				
				next_state, reward, terminates, truncates, info = self.env.step(action)

				action_num = np.array([action_num])
				self.replay_buffer.append(state, action_num, [reward/10], next_state, [int(terminates)])
				if self.now_retrain_step >= self.warmup_steps:
					self.update()

				self.total_time_step += 1
				self.now_retrain_step += 1 
				total_reward += reward
				state = next_state
				if terminates or truncates:
					progress = info['progress']   	###
					lap = int(info['lap'])  		###
					score = lap + progress - 1.   	###
					self.writer.add_scalar('Train/Episode Reward', score, self.total_time_step)
					print(
						'Step: {}\tEpisode: {}\twall_collision: {}\tLap: {:3d}\tProgress: {:.5f}\tTotal reward: {:.5f}\tscore: {:.5f}'
						.format(self.total_time_step, episode+1, info['wall_collision'], lap ,progress, total_reward, score))
				
					break
			
			if (episode+1) % self.eval_interval == 0:
				# save model checkpoint
				avg_score = self.evaluate()
				self.save(os.path.join(self.writer.log_dir, f"model_{self.total_time_step}_{int(avg_score*1000)}.pth"))
				self.writer.add_scalar('Evaluate/Episode Reward', avg_score, self.total_time_step)

	def evaluate(self):
		print("==============================================")
		print("Evaluating...")
		
		all_rewards = []
		for episode in range(self.eval_episode):
			total_reward = 0
			state, infos = self.test_env.reset()
			for t in range(10000):
				action,_ = self.decide_agent_actions(state)
				next_state, reward, terminates, truncates, info = self.test_env.step(action)
				total_reward += reward
				state = next_state
				if terminates or truncates:
					progress = info['progress']   	###
					lap = int(info['lap'])  		###
					score = lap + progress - 1.   	### 
					print(
						'Episode: {}\tProgress: {:.5f}\tavg score: {:.2f}'
						.format(episode+1, progress, score*1000))
					all_rewards.append(score)
					break

		avg = sum(all_rewards) / self.eval_episode
		print(f"average score: {avg}")
	
		#self.connect()
		print("==============================================")
		return avg
	
	# save model
	def save(self, save_path):
		torch.save(
				{
					'actor': self.actor_net.state_dict(),
					'critic1': self.critic_net1.state_dict(),
					'critic2': self.critic_net2.state_dict(),
				}, save_path)

	# load model
	def load(self, load_path):
		checkpoint = torch.load(load_path)
		self.actor_net.load_state_dict(checkpoint['actor'])
		self.critic_net1.load_state_dict(checkpoint['critic1'])
		self.critic_net2.load_state_dict(checkpoint['critic2'])

	# load model weights and evaluate
	def load_and_evaluate(self, load_path):
		self.load(load_path)
		self.evaluate()
	
	def record_video(self):
		frameStack = deque(maxlen=self.frames)
		images = []
		len = 0
		env = RaceEnv(
				scenario='austria_competition', # e.g., 'austria_competition', 'circle_cw_competition_collisionStop' 'austria_competition_collisionStop'
				render_mode='rgb_array_birds_eye',
				reset_when_collision=True, # Only work for 'austria_competition' and 'austria_competition_collisionStop'
			)

		obs, info = env.reset()
		while True:
			#obs = np.array(obs).astype(np.uint8)
			obs = obs.transpose((1, 2, 0))
			if len % 1 == 0:
				#obs = cv2.convertScaleAbs(obs, -100, 1.1)
				images.append(obs)
			state = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
			frameStack.append(state)
			state = np.stack(frameStack, axis=0)
			
			# Decide an action based on the observation (Replace this with your RL agent logic)
			if(len<self.frames):
				action = np.array([1. , 0.])
			else:
				action, _ = self.decide_agent_actions(state)

			next_obs, _, terminal, truncates, info = env.step(action)
			obs = next_obs

			if terminal:
				#record video
				progress = info['progress']   	###
				lap = int(info['lap'])  		###
				score = lap + progress - 1.   	### 
				cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
				video_name = f'results/{cur_time}_score{score:.4f}.mp4'
				Path(video_name).parent.mkdir(parents=True, exist_ok=True)
				height, width, layers = images[0].shape
				fourcc = cv2.VideoWriter_fourcc(*'mp4v')
				video = cv2.VideoWriter(video_name, fourcc, 30, (width, height))
				for image in images:
					video.write(image)
				cv2.destroyAllWindows()
				video.release()
				print(f'============ Terminal ============')
				if info.get('n_collision') is not None:
					print("Collision: ",info["n_collision"],"    collision_penalties: ", info['collision_penalties'])
				print(f'score: {score}, Video saved to {video_name}!')
				print(f'===================================')
				return
			len = len + 1
