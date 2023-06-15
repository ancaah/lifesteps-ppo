import numpy as np


class Memory():
    def __init__(self, num_envs, obs_shape, timesteps):

        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.timesteps = timesteps

        self._flattened = False

        self.reset()

    def reset(self):

        if self._flattened:
            del self.f_obs
            del self.f_returns
            del self.f_actions
            del self.f_rewards
            del self.f_probs
            del self.f_terminateds
            del self.f_truncateds
            del self.f_values
            del self.f_advantages

        timesteps = self.timesteps
        num_envs = self.num_envs
        obs_shape = self.obs_shape

        self.obss = np.zeros(shape=(timesteps, num_envs) + obs_shape)
        self.returns = np.zeros(shape=(timesteps, num_envs))
        self.actions = np.zeros(shape=(timesteps, num_envs))
        self.rewards = np.zeros(shape=(timesteps, num_envs))
        self.probs = np.zeros(shape=(timesteps, num_envs))
        self.terminateds = np.zeros(shape=(timesteps, num_envs))
        self.truncateds = np.zeros(shape=(timesteps, num_envs))
        self.values = np.zeros(shape=(timesteps, num_envs))
        self.advantages = np.zeros(shape=(timesteps, num_envs))


    def flatten(self):
        
        self._flattened = True

        timesteps = self.timesteps
        num_envs = self.num_envs
        obs_shape = self.obs_shape

        self.f_obs = self.obss.reshape(shape=(-1) + obs_shape)
        self.f_returns = self.returns.reshape(shape=(-1))
        self.f_actions = self.returns.reshape(shape=(-1))
        self.f_rewards = self.returns.reshape(shape=(-1))
        self.f_probs = self.returns.reshape(shape=(-1))
        self.f_terminateds = self.returns.reshape(shape=(-1))
        self.f_truncateds = self.returns.reshape(shape=(-1))
        self.f_values = self.returns.reshape(shape=(-1))
        self.f_advantages = self.returns.reshape(shape=(-1))

        #return f_obs, f_actions, f_rewards, f_terminateds, f_truncateds, f_probs, f_values, f_advantages, f_returns
