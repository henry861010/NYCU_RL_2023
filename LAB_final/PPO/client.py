import argparse
import json
import numpy as np
import requests
from ppo_agent_atari import AtariPPOAgent
from collections import deque
import cv2


def connect(agent, url: str = 'http://localhost:5000',frames=8):
    len = 0
    frameStack = deque(maxlen=frames)
    while True:
        # Get the observation
        response = requests.get(f'{url}')
        if json.loads(response.text).get('error'):
            print(json.loads(response.text)['error'])
            break
        obs = json.loads(response.text)['observation']
        obs = np.array(obs).astype(np.uint8)


        obs = obs.transpose((1, 2, 0))
        state = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        frameStack.append(state)
        obs = np.stack(frameStack, axis=0)

        if len<frames:
            action_to_take = np.array([1. , 0.])
        else:
            action_to_take, _, _, _, _ = agent.decide_agent_actions(obs)  
        len =len + 1

        # Send an action and receive new observation, reward, and done status
        response = requests.post(f'{url}', json={'action': action_to_take.tolist()})
        if json.loads(response.text).get('error'):
            print(json.loads(response.text)['error'])
            break

        result = json.loads(response.text)
        terminal = result['terminal']

        if terminal:
            print('Episode finished.')
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str, default='http://localhost:5000', help='The url of the server.')
    args = parser.parse_args()

    config = {
		"gpu": True,
		"training_steps": 1e10,
		"update_sample_count": 20000,#10000
		"discount_factor_gamma": 0.99,
		"discount_factor_lambda": 0.95,
		"clip_epsilon": 0.2,
		"max_gradient_norm": 0.5,
		"batch_size": 2048,
		"logdir": 'log/Enduro/',
		"update_ppo_epoch": 10,  #3
		"learning_rate": 2.5e-4,
		"value_coefficient": 0.5,
		"entropy_coefficient": 0.01,
		"horizon": 2048,
		"eval_interval": 70,
		"eval_episode": 5,  #####
		"action_space": 15,
		"frames_stack": 8,
		"total_time_step": 0
	}

    actions = [
		[-1, 1.0],[0, 1.0],[1, 1.0],
        [-1, 0.4],[0, 0.4],[1, 0.4],
		[-1, 0.0],[0, 0.0],[1, 0.0],
		[-1,-0.4],[0,-0.4],[1,-0.4],
        [-1,-1.0],[0,-1.0],[1,-1.0]
	]

    rand_agent = AtariPPOAgent(config, actions)
    rand_agent.load('.\\log\\Enduro\\aus_model_4330201_550.pth')
    connect(rand_agent, url=args.url, frames=int(config['frames_stack']))
    #rand_agent.record_video()
