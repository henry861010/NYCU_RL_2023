import torch
import torch.nn as nn
import numpy as np
import os
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from replay_buffer.gae_replay_buffer import GaeSampleMemory
from replay_buffer.replay_buffer import ReplayMemory
from abc import ABC, abstractmethod
import cv2
from datetime import datetime
from environment_wrapper.final_project_env.racecar_gym.env import RaceEnv
from pathlib import Path
from tutor import tutor

class PPOBaseAgent(ABC):
	def __init__(self, config):
		self.gpu = config["gpu"]
		self.device = torch.device("cuda" if self.gpu and torch.cuda.is_available() else "cpu")
		self.total_time_step = int(config["total_time_step"])
		self.training_steps = int(config["training_steps"])
		self.update_sample_count = int(config["update_sample_count"])
		self.discount_factor_gamma = config["discount_factor_gamma"]
		self.discount_factor_lambda = config["discount_factor_lambda"]
		self.clip_epsilon = config["clip_epsilon"]
		self.max_gradient_norm = config["max_gradient_norm"]
		self.batch_size = int(config["batch_size"])
		self.value_coefficient = config["value_coefficient"]
		self.entropy_coefficient = config["entropy_coefficient"]
		self.eval_interval = config["eval_interval"]
		self.eval_episode = config["eval_episode"]

		self.gae_replay_buffer = GaeSampleMemory({
			"horizon" : config["horizon"],
			"use_return_as_advantage": False,
			"agent_count": 1,
			})

		self.writer = SummaryWriter(config["logdir"])
		self.tutor = tutor()

	@abstractmethod
	def decide_agent_actions(self, observation):

		return NotImplementedError

	@abstractmethod
	def update(self):
		# sample a minibatch of transitions
		batches = self.gae_replay_buffer.extract_batch(self.discount_factor_gamma, self.discount_factor_lambda)
		# calculate the loss and update the behavior network

		return NotImplementedError
	



	def train(self):
		episode_idx = 0
		while self.total_time_step <= self.training_steps:
			observation, info = self.env.reset()
			#cv2.imwrite('123.png',observation[0])
			episode_reward = 0
			episode_len = 0
			episode_idx += 1
			while info['progress']<0.33:
				action_env, action, logp_pi, value, entropy = self.tutor.decide_agent_actions(observation=observation)
				next_observation, reward, terminate, truncate, info = self.env.step(action_env)
				observation = next_observation
			while True:
				action_env, action, logp_pi, value, entropy = self.decide_agent_actions(observation)
				next_observation, reward, terminate, truncate, info = self.env.step(action_env)
				# observation must be dict before storing into gae_replay_buffer
				# dimension of reward, value, logp_pi, done must be the same
				obs = {}
				obs["observation_2d"] = np.asarray(observation, dtype=np.float32)
				self.gae_replay_buffer.append(0, {
						"observation": obs,    # shape = (4,84,84)
						"action": action,      # shape = (1,)
						"reward": reward,      # shape = ()
						"value": value,        # shape = ()
						"logp_pi": logp_pi,    # shape = ()
						"done": terminate,     # shape = ()
					})
				
				# update the weight each update_sample_count, to prevent there exsit large different between current and old policy
				if len(self.gae_replay_buffer) >= self.update_sample_count:
					self.update()
					self.gae_replay_buffer.clear_buffer()

				episode_reward += reward
				episode_len += 1
				self.total_time_step += 1
				
				if terminate or truncate:
					progress = info['progress']   	###
					lap = int(info['lap'])  		###
					score = lap + progress - 1.   	###
					self.writer.add_scalar('Train/Episode score', score, self.total_time_step)
					print(
						'Step: {}\tEpisode: {}\twall_collision: {}\tLap: {:3d}\tProgress: {:.5f}\tTotal reward: {:.5f}\tscore: {:.5f}'
						.format(self.total_time_step,episode_idx, info['wall_collision'], lap ,progress, episode_reward, score))
				
					break
				observation = next_observation
				
			if episode_idx % self.eval_interval == 0:
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
				if infos['progress']<0.33:
					action_env, action, logp_pi, value, entropy = self.tutor.decide_agent_actions(observation=state)
					next_state, reward, terminates, truncates, info = self.test_env.step(action_env)
				else:
					action_env, action, logp_pi, value, entropy = self.decide_agent_actions(state)
					next_state, reward, terminates, truncates, info = self.test_env.step(action_env)
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
		torch.save(self.net.state_dict(), save_path)

	# load model
	def load(self, load_path):
		self.net.load_state_dict(torch.load(load_path))

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
				action, _, _, _, _ = self.decide_agent_actions(state)

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
	



	

