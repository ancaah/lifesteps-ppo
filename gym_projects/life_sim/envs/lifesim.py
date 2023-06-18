'''My environment class'''
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
import numpy as np


class LifeSim(Env):
    '''Life Simulation Environment for the Autonomous Adaptive Systems'''

    metadata = {"render_modes": ["text"]}

    def __init__(self, render_mode=None, max_timesteps = 300):
        # define your environment
        # action space, observation space

        self._max_timesteps = max_timesteps

        #self.terminated = False
        #self.truncated = False
        # Money, Health,
        # Work Development, Social Development
        self.observation_space = Box(
            low=0, high=10, shape=(4,), dtype=np.float32)
        self.obs_dict = {"money": 0, "health": 1,
                         "work_development": 2, "social_development": 3}

        # Work, Sport, Sociality
        self.action_space = Discrete(3)

        # current state
        # self.state = np.array([3., 3., 1., 1.], dtype=np.float32)
        self._decreasing_func = np.array([-0.5, -0.3, -0.07, -0.1])
        self.min_money = 0
        self.min_health = 0
        self._current_timestep = 0

        # reward collected
        self.collected_reward = 0
        # reward lambda
        self.l_1 = 2
        self.l_2 = 0.5


        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self._action_outcome_mapping = {
            # Work
            0: [1, 0, 0.3, 0],
            # Sport
            1: [0, 1, 0, 0],
            # Sociality
            2: [0, 0, 0, 0.4],
        }
        self._action_names = {
            0: "work",
            1: "sport",
            2: "sociality"
        }

        self._last_action = None

    def accumulate_reward(self):
        '''accumulate the total reward using this method'''
        return self.l_1*(self.state[self.obs_dict["work_development"]]
                         + self.state[self.obs_dict["social_development"]])

    def _get_obs(self):
        return self.state

    def _get_info(self):
        return {
            "last_action": self._last_action,
            "last_reward": self._last_reward,
            "last_state": self._last_state
            }
    
    def _set_info(self, state, action, reward):
        self._last_action = action
        self._last_reward = reward
        self._last_state = state

    def step(self, action):
        # take some action
        #self._last_action = action
        outcome = self._action_outcome_mapping[action]
        self._current_timestep += 1

        # update state with decreasing function of Work and Social Development
        new_state = np.add(self.state, self._decreasing_func, dtype=np.float32)
        new_state = np.clip(new_state, a_min=0, a_max=10)
        new_state = np.concatenate([[new_state[0], new_state[1]], np.clip([new_state[2], new_state[3]], a_min=0, a_max=10)])

        # update state with action's outcome
        new_state = np.add(new_state, outcome, dtype=np.float32)
        new_state = np.clip(new_state, a_min=0, a_max=10)

        self.state = new_state

        observation = self._get_obs()
        info = self._get_info()

        #self.collected_reward += self.accumulate_reward()
        self.collected_reward = self.accumulate_reward()

        truncated = True if self._current_timestep == self._max_timesteps else False
        
        terminated = (self.state[self.obs_dict["money"]] == 0 or self.state[self.obs_dict["health"]] == 0)
        
        reward = self.collected_reward if truncated else 0
        reward = -10+self.collected_reward if terminated else 0

        self._set_info(new_state, action, reward)

        #self.render()

        return observation, reward, terminated, truncated, info

    def render(self):
        # render your environment (can be a visualisation/print)
        #result = None
        if self.render_mode == "text":
            #print(self._get_info().get('last_action'))
            #print(self._action_names.get(0))
            print(self._render_text())
            
        #return result

    def _render_text(self):
        '''Returns a text representation of the state'''
        i = self._get_info()
        if i.get('last_action') < 0:
            return f"State -> {i['last_state']}"
        else:
            return f"State -> {i['last_state']}  |  Reward: {i['last_reward']}  |  Last Action: {self._action_names[i['last_action']]}  |"
        #return f"State -> {i['last_state']}  |  Reward: {i['last_reward']}  |  Last Action:  |"

    def reset(self, seed=None, options=None):
#    def reset(self, seed=None, options=None):
        # reset your environment
        super().reset(seed=seed)

        np.random.seed(seed)
        space = np.linspace(2.5, 3.5, 50)

        self.state = np.array([space[np.random.randint(0, len(space))],
                               space[np.random.randint(0, len(space))],
                               space[np.random.randint(0, len(space))] / 1.5,
                               space[np.random.randint(0, len(space))] / 1.5], dtype=np.float32)
        self.collected_reward = 0
        self._current_timestep = 0
        #self._max_timesteps = max_timesteps

        observation = self._get_obs()
        
        self._set_info(self.state, -1, -1)
        info = self._get_info()

        #if self.render_mode is not None:
        #    self.render(self.render_mode)

        return observation, info
