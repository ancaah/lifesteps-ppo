import numpy as np

class Memory():
    def __init__(self, num_envs, obs_shape, timesteps):
        
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.timesteps = timesteps
        self.reset()

    def reset(self):
        
        timesteps = self.timesteps
        num_envs = self.num_envs
        obs_shape = self.obs_shape

        self.obss = np.zeros(shape=(timesteps,num_envs) + obs_shape)
        self.returns = np.zeros(shape=(timesteps,num_envs))
        self.actions = np.zeros(shape=(timesteps,num_envs))
        self.rewards = np.zeros(shape=(timesteps,num_envs))
        self.probs = np.zeros(shape=(timesteps,num_envs))
        self.terminateds = np.zeros(shape=(timesteps,num_envs))
        self.truncateds = np.zeros(shape=(timesteps,num_envs))
        self.values = np.zeros(shape=(timesteps,num_envs))
        self.advantages = np.zeros(shape=(timesteps,num_envs))
        

