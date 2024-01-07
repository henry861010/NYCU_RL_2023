from typing import Union

import numpy as np

import gym
from gym.error import DependencyNotInstalled
from gym.spaces import Box


class GetPartObservation(gym.ObservationWrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)
        shape = (150, 150)
        self.shape = tuple(shape)

        obs_shape = self.shape + env.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        observation = observation[0:150, 10:160]
        return observation