import argparse
from collections import deque
import itertools
import random
import time
import cv2

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
#from final_project_env.racecar_gym.env import RaceEnv
from racecar_gym.env import RaceEnv



class RaceCarEnvironment:
	def __init__(self, N_frame=4, test=False):
		self.test = test
		if not self.test:
			self.env = RaceEnv(
				scenario='austria_competition_collisionStop', # e.g., 'austria_competition', 'circle_cw_competition_collisionStop' 'austria_competition_collisionStop'
				render_mode='rgb_array_birds_eye',
				reset_when_collision=False, # Only work for 'austria_competition' and 'austria_competition_collisionStop'
			)
		else:
			self.env = RaceEnv(
				scenario='austria_competition', # e.g., 'austria_competition', 'circle_cw_competition_collisionStop' 'austria_competition_collisionStop'
				render_mode='rgb_array_birds_eye',
				reset_when_collision=True, # Only work for 'austria_competition' and 'austria_competition_collisionStop'
			)
		self.action_space = self.env.action_space
		self.observation_space = self.env.observation_space
		self.ep_len = 0
		self.frames = deque(maxlen=N_frame)

		self.obstacle_rate_2 = -0.8333333333
		self.obstacle_rate_1 = 2.8333333333
		self.obstacle_rate_0 = -1
		self.reward_rate = 1500
		self.reward_bias = 0.6
	
	def check_car_position(self, obs):
		# cut the image to get the part where the car is
		part_image = obs[60:84, 40:60, :]

		road_color_lower = np.array([90, 90, 90], dtype=np.uint8)
		road_color_upper = np.array([120, 120, 120], dtype=np.uint8)
		grass_color_lower = np.array([90, 180, 90], dtype=np.uint8)
		grass_color_upper = np.array([120, 255, 120], dtype=np.uint8)
		road_mask = cv2.inRange(part_image, road_color_lower, road_color_upper)
		grass_mask = cv2.inRange(part_image, grass_color_lower, grass_color_upper)
		# count the number of pixels in the road and grass
		road_pixel_count = cv2.countNonZero(road_mask)
		grass_pixel_count = cv2.countNonZero(grass_mask)

		# save image for debugging
		# filename = "images/image" + str(self.ep_len) + ".jpg"
		# cv2.imwrite(filename, part_image)

		return road_pixel_count, grass_pixel_count

	def step(self, action):
		obs, reward, terminates, truncates, info = self.env.step(action)
		obs = obs.transpose((1, 2, 0))  # (3,128,128) to (128,128,3)
		original_reward = reward
		original_terminates = terminates
		self.ep_len += 1

		#print("reward1: ",reward)
		if reward==0:
			obstacle_penalty = self.obstacle_rate_2*info['obstacle']**2 + self.obstacle_rate_1*info['obstacle'] + self.obstacle_rate_0
			reward = self.reward_rate * reward - self.reward_bias
			reward = reward + obstacle_penalty * 0.4

		if terminates:
			reward = -20

		#print("penalty: ",obstacle_penalty)
		#print("obstacle: ",info['obstacle'])
		#print("reward2: ",reward)
		#print("  ")

		#print("v: ",self.velocity_pre)
		#print("action: ",action)
		#print(self.progress_pre)
		#print(progress_now)
		#print("reward: ",reward)
		#print("obstacle: ",info['obstacle'],"   reward: ",reward)
		#print("terminates: ",terminates,"   info['wall_collision']: ",info['wall_collision'])
		#print("  ")

		# convert to grayscale
		obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY) # 96x96

		# save image for debugging
		# filename = "images/image" + str(self.ep_len) + ".jpg"
		# cv2.imwrite(filename, obs)

		# frame stacking
		self.frames.append(obs)
		obs = np.stack(self.frames, axis=0)

		if self.test:
			reward = original_reward

		return obs, reward, terminates, truncates, info
	
	def reset(self, *args, **kwargs):
		if not self.test:
			kwargs['options'] = {'mode':'random'} 
		obs, info = self.env.reset(**kwargs)
		#print("--------------------------------------------------")
		#print(info)
		#print("--------------------------------------------------")
		self.ep_len = 0
		self.progress_pre = info['progress']
		obs = obs.transpose((1, 2, 0))  # (3,128,128) to (128,128,3)
		obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY) # 96x96

		# frame stacking
		for _ in range(self.frames.maxlen):
			self.frames.append(obs)
		obs = np.stack(self.frames, axis=0)

		return obs, info
	
	def render(self):
		self.env.render()
	
	def close(self):
		self.env.close()

if __name__ == '__main__':
	env = RaceCarEnvironment(test=True)
	obs, info = env.reset()
	done = False
	total_reward = 0
	total_length = 0
	t = 0
	while not done:
		t += 1
		action = env.action_space.sample()
		action[2] = 0.0
		obs, reward, terminates, truncates, info = env.step(action)
		print(f'{t}: road_pixel_count: {info["road_pixel_count"]}, grass_pixel_count: {info["grass_pixel_count"]}, reward: {reward}')
		total_reward += reward
		total_length += 1
		env.render()
		if terminates or truncates:
			done = True

	print("Total reward: ", total_reward)
	print("Total length: ", total_length)
	env.close()
