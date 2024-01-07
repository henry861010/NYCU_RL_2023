import gymnasium as gym
from gymnasium import spaces
from racecar_gym.env import RaceEnv
from gymnasium.spaces import Box
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import  SubprocVecEnv
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper
#from environment_wrapper.racecarEnv import RaceCarEnvironment
# gymenv modify - https://www.gymlibrary.dev/content/environment_creation/
# reload model - https://stackoverflow.com/questions/73737008/stable-baselines3-ppo-model-loaded-but-not-working
# ppo - readthedocs



class discreteEnv(RaceEnv):
    def __init__(self):
        super().__init__(scenario='austria_competition_collisionStop', render_mode='rgb_array_birds_eye',reset_when_collision=False)
        #super().__init__(scenario='austria_competition', render_mode='rgb_array_birds_eye',reset_when_collision=True)
        self.action_space = spaces.Discrete(6)
        self.action_env = [
            [0.8,1],[1,0],[0.8,-1],[-0.6,0],[0,1],[0,-1]
        ]
        self.observation_space = Box(low=0, high=255, shape=(1, 100, 100), dtype=np.uint8 )

    def step(self, action_num):
        actions =  self.action_env[action_num]
        obs, rew, terminated, truncated, info = super().step(actions)
        obs = np.mean( obs, axis=0, keepdims=True)
        obs = obs[:,14:114,14:114]
        return obs, rew, terminated, truncated, info
        
    def reset(self, seed=None, options=None):
        obs, info = super().reset(options=dict(mode='random'),seed=seed)
        obs = np.mean( obs, axis=0, keepdims=True)
        obs = obs[:,14:114,14:114]
        return obs, info



def make_environment():
    def _init():
        env = discreteEnv()
        return env
    return _init
'''
def evaluate(model):
    print("==============================================")
    print("Evaluating...")
    test_env= [make_environment() for seed in range(1)]
    test_env = SubprocVecEnv(test_env)
    test_env = VecFrameStack(test_env, 8)
    
    all_rewards = []
    for episode in range(5):
        score = 0
        state = test_env.reset()
        for t in range(10000):
            action, _ = model.predict(state)
            next_state, reward, terminates,  info = test_env.step(action)
            score += reward
            state = next_state
            if terminates:
                print("round-",episode,"   progress: ",score)
                all_rewards.append(score)
                break

    avg = sum(all_rewards) / 5
    print(f"average score: {avg}")

    #self.connect()
    print("==============================================")
    return avg
    '''


if __name__ == '__main__':
 
    envs= [make_environment() for seed in range(20)]
    envs = SubprocVecEnv(envs)
    envs = VecFrameStack(envs, 8)

    model = PPO("CnnPolicy", envs, verbose=1,n_steps=2048,ent_coef=0.35,batch_size=128)
    model.load("ppo-baseline-agent", env=envs)

    count = 0
    while True:
        model.learn(total_timesteps=100)
        model.save("ppo-baseline-agent")
        print("finish-",count," !!!")
        count = count + 1
    
    #model = PPO.load("ppo-baseline-agent")

    #evaluate(model)

