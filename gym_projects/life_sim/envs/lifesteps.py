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
        
        # global values used for indexing
        money = 0
        health = 1
        social = 2

        # check if render mode is legit
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        # check if gamemode is legit
        assert gamemode in self.metadata["game_modes"]
        self.gamemode = gamemode
        
        self.difficulty = difficulty

        # set the observation space
        # money, health, sociality, friends
        self.arrtype = np.int16
        self.observation_space = spaces.Box(low=np.zeros((4,)), high=np.array([100,100,100,1]), shape=(4,), dtype=self.arrtype)


        # set the action space
        # work, sport, social
        self.action_space = spaces.Discrete(3)

        # this defines the decrease on life's values at each timestep
        self._standard_deficit = np.array([-5, -3, -1], dtype=self.arrtype)
        self._deficit = self._standard_deficit
        
        #if gamemode == 'monopoly':
        #    self._trouble_prob_inc = 0.05

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
        # only the 'text' render mode is supported by LifeSteps
        if self.render_mode == "text":
            print(self._render_text())
        

    def _render_text(self):
        '''Returns a text representation of the state'''

        i = self._get_info()
        str = f"| Life ->\t{i.get('life')}  -   {'F: many!!!' if i.get('friends') else 'F: alone..'}"
        if i.get('last_action') < 0:
            str = f"{str}, I just started playing!"
        else:
            str = f"{str}\t\t|  Last Action: {self.action_names[i['last_action']]}"
            

            if self.gamemode == 'monopoly' and i.get('trouble') != -1:
                if i.get('friends'):
                    f = np.random.choice(['Piz', 'Leandro', 'Dave', 'Dudu'])

                p = i.get('points_loss')
                str = f"{str}\t\t| {i.get('trouble')}{f'but {f} helped,' if i.get('friends') and p < 0 else ''} {f'{p} points loss' if p <= 0 else ''}"

            if i.get('done') == True:
                str = f"{str}\n\n--! Game finished with reward {i['last_reward']} !--\n"
        return str


    # support function for calculating the reward
    # calculate difference between the difficulty level and the minimum life's value
    def distance_from_target(self):
        eval = np.min(self._life)
        return self._target - eval


    # function for calculating the reward if the episode reached the maximum number of steps
    def calculate_reward(self):
        d = self.distance_from_target()
        return -d if d > 0 else 1

    # performs an environment's step
    def step(self, action):

        self._current_timestep += 1

        # STATE UPDATE
        # firstly adding the action outcome to the _life array
        # then subtracting the decreasing function
        outcome = self._action_outcome_mapping[action]

        self._life += outcome        

        # if in monopoly gamemode, the behaviour is different
        if self.gamemode == 'monopoly':

            # check if in this timestep the player "picks a chance card"
            trouble = np.random.choice([False, True], p=[1 - self._trouble_probability, self._trouble_probability])
            
            # if trouble is True, the player encountered some type of trouble
            if trouble == True:

                # resetting the probability of getting a chance card to zero
                self._trouble_probability = 0

                # select the type of penalty -> money or health
                trouble_type = np.random.choice([money, health])
                trouble_deficit = np.zeros(shape=(3,), dtype=self.arrtype)
                trouble_deficit[trouble_type] = -10 if self._friends == 0 else -4 + np.random.randint(4)

                # apply the penalty
                self._life += trouble_deficit

                # update the info
                self._set_info(None, None, None, None, None, trouble_type, np.min(trouble_deficit))
            else:
                self._set_info()    # sets all info to their default value

                # increment the probability of getting a "chance card"
                self._trouble_probability += self._trouble_prob_inc
                self._trouble_probability = min(self._trouble_probability, 1)
                    
        # apply the usual timestep's decrease on life points
        self._life += self._standard_deficit

        # friends = 1 if the social development of the player goes above 40
        # or if he already has friends
        self._friends = 1 if self._life[social] > 40 else self._friends
        
        # clip the state to avoid going outside the observation space
        self._life = np.clip(self._life, a_min=0, a_max=100, dtype=self.arrtype)

        observation = self._get_obs()

        # truncation is ture if the episode reached its max length, so it's stopped
        truncated = True if self._current_timestep == self.max_timesteps else False

        # terminations is true if that the player died
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
        
        # setting default probability and probability increase for the monopoly gamemode
        if self.gamemode == 'monopoly':
            self._trouble_probability = 0
            self._trouble_prob_inc = 0.03
            #self._trouble_prob_inc = np.random.choice([0.01, 0.03, 0.08])

        # sample the initial state
        s_life = np.arange(25, 35, 1)
        money, health, social = np.random.choice(s_life, 3)
        
        # set the initial state
        self._life = np.array([money, health, social], dtype=self.arrtype)
        self._friends = 0
        self._target = self.difficulty

        self._current_timestep = 0

        observation = self._get_obs()
        
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
                "friends": self.i_friends
            }

        if self.gamemode == 'monopoly':
            d['trouble'] = lost[self.i_trouble_t]
            d['points_loss'] = self.i_points_loss

        return d


    def _set_info(self, life = None, friends = None, action = None, reward = None, done = None, trouble = -1, points_loss = 1):
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