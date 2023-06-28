'''My environment class'''
from gymnasium import Env
from gymnasium import spaces
import numpy as np


class LifeSteps(Env):
    '''Life Simulation Environment for the Autonomous Adaptive Systems'''

    metadata = {"render_modes": ["text", "text_summary"], "render_fps": 1, "game_modes": ["standard", "monopoly"]}    

    def __init__(self, render_mode=None, max_timesteps = 300, difficulty = 0, gamemode = 'standard'):
        # define your environment
        # action space, observation space
        
        global money
        global health
        global social
        
        money = 0
        health = 1
        social = 2

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        assert gamemode in self.metadata["game_modes"]
        self.gamemode = gamemode
        
        self.difficulty = difficulty

        # Money, Health, Social Development
        '''
        self.observation_space = spaces.Dict(
            {
            "life": spaces.Box(low=0, high=100, shape=(3,), dtype=int),
            "friends": spaces.Discrete(1),
            "target": spaces.Discrete(1)
            }
        )'''

        self.arrtype = np.int16
        self.observation_space = spaces.Box(low=np.zeros((4,)), high=np.array([100,100,100,1]), shape=(4,), dtype=self.arrtype)
#        self.observation_space = spaces.Box([spaces.Box(low=0, high=100, shape=(3,), dtype=self.arrtype),
#                                               spaces.Box(low=0, high=1, dtype=self.arrtype)])

        # Work, Sport, Sociality
        self.action_space = spaces.Discrete(3)

        self._decreasing_func = np.array([-5, -3, -1], dtype=self.arrtype)
        
        # constraints
        self.min_money = 0
        self.min_health = 0
        self.max_timesteps = max_timesteps


        # useful dictionaries to ease the reading of the code
        self.life_dict = {"money": 0, "health": 1, "social": 2}

        self._action_outcome_mapping = {
            # Work
            0: np.array([10, 0, 0], dtype=self.arrtype),
            # Sport
            1: np.array([0, 10, 0], dtype=self.arrtype),
            # Sociality
            2: np.array([0, 0, 10], dtype=self.arrtype)
        }
        self.action_names = {
            0: "work",
            1: "sport",
            2: "sociality"
        }

    def render(self):
        # render your environment (can be a visualisation/print)
        if self.render_mode == "text":
            print(self._render_text())
        
    def _render_text(self):
        '''Returns a text representation of the state'''
        i = self._get_info()
        if i.get('last_action') < 0:
            str = f"| State -> {i['last_state']}, I just started playing!"
        else:
            str = f"| State -> {i['last_state']}  |  Last Action: {self.action_names[i['last_action']]} |"
            
            if i.get('done') == True:
                str = f"{str}\n--! Game finished with reward {i['last_reward']} !--\n"
        return str

    def distance_from_target(self):
        eval = np.min(self._life)
        return self._target - eval

    def calculate_reward(self):
        d = self.distance_from_target()
        return -d if d > 0 else 1

    def step(self, action):

        self._current_timestep += 1
        
        # STATE UPDATE
        # firstly adding the action outcome to the _life array
        # then subtracting the decreasing function
        
        outcome = self._action_outcome_mapping[action]
        
        self._life += outcome
        self._life += self._decreasing_func

        # friends = 1 if the social development of the player goes upper than 50
        # or if he already has friends
        self._friends = 1 if self._life[social] > 50 else self._friends
        self._life = np.clip(self._life, a_min=0, a_max=100, dtype=self.arrtype)

        observation = self._get_obs()
        info = self._get_info()

        truncated = True if self._current_timestep == self.max_timesteps else False
        terminated = (self._life[money] == 0 or self._life[health] == 0)
        
        reward = 0
        if terminated:
            reward = self._current_timestep - self.max_timesteps - self.difficulty
        if truncated:
            reward = self.calculate_reward()

        self._set_info(self._life, action, reward, truncated or terminated)

        return observation, reward, terminated, truncated, info


    def reset(self, seed=None, options=None):

        # reset your environment
        super().reset(seed=seed)

        np.random.seed(seed)
        s_life = np.arange(25, 35, 1)
        
        money, health, social = np.random.choice(s_life, 3)
        
        self._life = np.array([money, health, social], dtype=self.arrtype)
        self._friends = 0
        self._target = self.difficulty

        self._current_timestep = 0

        observation = self._get_obs()
        
        self._set_info(self._life, -1, -1, -1)
        info = self._get_info()

        if self.render_mode == 'text':
            self.render()

        return observation, info


    def _get_obs(self):
        return np.array([
                        self._life[0],
                        self._life[1],
                        self._life[2], 
                        self._friends], dtype=self.arrtype)

    '''def _get_obs(self):
        return {
            'life': self._life,
            'friends': self._friends,
            'target': self._target
        }
    '''

    def _get_info(self):
        return {
            "done": self._done,
            "last_action": self._last_action,
            "last_reward": self._last_reward,
            "last_state": self._last_state
            }

    def _set_info(self, state, action, reward, done):
        self._last_state = state
        self._last_action = action
        self._last_reward = reward
        self._done = done