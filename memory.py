import numpy as np

# the general idea of separating the memory instance from the agent to have a cleaner code was taken by
# https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/PPO/tf2/memory.py

class Memory():
    def __init__(self, num_envs, obs_shape, timesteps):

        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.timesteps = timesteps

        self._flattened = False

        self.reset()

    def reset(self):

        if self._flattened:
            self._flattened = False
            del self.f_obss
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

        self.obss = np.zeros(shape=(timesteps, num_envs) + obs_shape, dtype=np.float32)
        
        self.actions = np.zeros(shape=(timesteps, num_envs), dtype=np.int32)
        self.rewards = np.zeros(shape=(timesteps, num_envs), dtype=np.float32)
        self.probs = np.zeros(shape=(timesteps, num_envs), dtype=np.float32)
        self.terminateds = np.zeros(shape=(timesteps, num_envs), dtype=np.float32)
        self.truncateds = np.zeros(shape=(timesteps, num_envs), dtype=np.float32)
        self.values = np.zeros(shape=(timesteps, num_envs), dtype=np.float32)

        self.advantages = np.zeros(shape=(timesteps, num_envs), dtype=np.float32)
        self.returns = np.zeros(shape=(timesteps, num_envs), dtype=np.float32)


    def flatten(self):
        '''Creates, inside the Memory object, a flattened instance of the np.ndarrays already initialized'''
        self._flattened = True

        obs_shape = self.obs_shape

        #self.f_obss = self.obss.reshape((-1,) + obs_shape)
        self.f_obss = self.obss.reshape((-1,) + obs_shape)
        
        self.f_actions = self.actions.reshape((-1,))
        self.f_rewards = self.rewards.reshape((-1,))
        self.f_probs = self.probs.reshape((-1,))
        self.f_terminateds = self.terminateds.reshape((-1,))
        self.f_truncateds = self.truncateds.reshape((-1,))
        self.f_values = self.values.reshape((-1,))
        
        self.f_returns = self.returns.reshape((-1,))
        self.f_advantages = self.advantages.reshape((-1,))