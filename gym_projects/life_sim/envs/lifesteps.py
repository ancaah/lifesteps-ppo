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


        self.arrtype = np.int16
        self.observation_space = spaces.Box(low=np.zeros((4,)), high=np.array([100,100,100,1]), shape=(4,), dtype=self.arrtype)


        # Work, Sport, Sociality
        self.action_space = spaces.Discrete(3)

        self._standard_deficit = np.array([-5, -3, -1], dtype=self.arrtype)
        self._deficit = self._standard_deficit
        
        if gamemode == 'monopoly':
            self._trouble_prob_inc = 0.05

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
            2: "social"
        }

    def render(self):
        # render your environment (can be a visualisation/print)
        if self.render_mode == "text":
            print(self._render_text())
        
    def _render_text(self):
        '''Returns a text representation of the state'''
        i = self._get_info()
        str = f"| Life ->\t{i['life']}  -   {i['friends']}"
        if i.get('last_action') < 0:
            str = f"{str}, I just started playing!"
        else:
            str = f"{str}\t\t|  Last Action: {self.action_names[i['last_action']]}"
            

            if self.gamemode == 'monopoly' and i.get('trouble') != -1:
                str = f"{str}\t\t| {i.get('trouble')} {i.get('points_loss')} points loss"

            if i.get('done') == True:
                str = f"{str}\n\n--! Game finished with reward {i['last_reward']} !--\n"
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

        if self.gamemode == 'monopoly':
            trouble = np.random.choice([False, True], p=[1 - self._trouble_probability, self._trouble_probability])
            if trouble == True:
                self._trouble_probability = 0

                trouble_type = np.random.choice([money, health])
                trouble_deficit = np.zeros(shape=(3,), dtype=self.arrtype)
                trouble_deficit[trouble_type] = -10 if self._friends == 0 else -10 + np.random.random_integers(8)

                self._life += trouble_deficit

                self._set_info(None, None, None, None, None, trouble_type, np.min(trouble_deficit))
            else:
                self._set_info()    # sets all info to None
                self._trouble_probability += self._trouble_prob_inc
                self._trouble_probability = min(self._trouble_probability, 1)
                    
        self._life += self._standard_deficit

        # friends = 1 if the social development of the player goes upper than 50
        # or if he already has friends
        self._friends = 1 if self._life[social] > 50 else self._friends
        self._life = np.clip(self._life, a_min=0, a_max=100, dtype=self.arrtype)

        observation = self._get_obs()

        truncated = True if self._current_timestep == self.max_timesteps else False
        terminated = (self._life[money] == 0 or self._life[health] == 0)
        
        reward = 0
        if terminated:
            reward = self._current_timestep - self.max_timesteps - self.difficulty
        if truncated:
            reward = self.calculate_reward()


        self._set_info(self._life, self._friends, action, reward, truncated or terminated, self.i_trouble_t, self.i_points_loss)
        info = self._get_info()

        return observation, reward, terminated, truncated, info


    def reset(self, seed=None, options=None):

        # reset your environment
        super().reset(seed=seed)

        np.random.seed(seed)
        
        if self.gamemode == 'monopoly':
            self._trouble_probability = 0
            self._trouble_prob_inc = np.random.choice([0.01, 0.03, 0.08])

        s_life = np.arange(25, 35, 1)
        
        money, health, social = np.random.choice(s_life, 3)
        
        self._life = np.array([money, health, social], dtype=self.arrtype)
        self._friends = 0
        self._target = self.difficulty

        self._current_timestep = 0

        observation = self._get_obs()
        
        #self._set_info(self._life, -1, -1, -1)
        self._set_info(self._life, self._friends, -1, -1, False)
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

    def _get_info(self):

        lost = {
            -1: "all ok",
            money:  "...you lost some pennies....",
            health: "...got hurt while cooking..."
        }

        d = {
                "done": self.i_done,
                "last_action": self.i_action,
                "last_reward": self.i_reward,
                "life": f"M: {self.i_life[money]},   H: {self.i_life[health]},   S: {self.i_life[social]}",
                "friends": "F: alone.." if self.i_friends == 0 else "F: many!!!"
            }

        if self.gamemode == 'monopoly':
            d['trouble'] = lost[self.i_trouble_t]
            d['points_loss'] = self.i_points_loss

        return d


    def _set_info(self, life = None, friends = None, action = None, reward = None, done = None, trouble = -1, points_loss = -1):
        self.i_done = self.i_done if done is None else done
        self.i_action = self.i_action if action is None else action
        self.i_reward = self.i_reward if reward is None else reward
        self.i_life = self.i_life if life is None else life
        self.i_friends = self.i_friends if friends is None else friends
        
        if self.gamemode == 'monopoly':
            self.i_trouble_t = trouble
            self.i_points_loss = points_loss

        elif self.gamemode == 'standard':
            self.i_trouble_t = None
            self.i_points_loss = None