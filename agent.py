'''this file provides the Agent abstract class and some of its useful implementations'''
from abc import ABC
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.initializers import Orthogonal
import numpy as np
import tensorflow_probability as tfp
import gymnasium as gym
from tqdm.notebook import tqdm
import memory
import utils
#from memory_profiler import profile


class Agent(ABC):
    def __init__(self):
        pass

    def predict(self, *args, **kwargs):
        pass

    def train(self, *args, **kwargs):
        pass

    def _calc_loss(self, *args, **kwargs):
        pass

    def play_one_step(self, *args, **kwargs):
        pass

    def play_episode(self, *args, **kwargs):
        pass

    def _policy(self, *args, **kwargs):
        pass

class PPO_Agent(Agent):
    
    
    def __init__(self, input_shape, n_outputs, gamma, lmbda, epsilon, c2, lr_actor, lr_critic, log = False, log_dir = None, env_max_timesteps=300, init_personal = True, units=32, verbose=0):

        self._verbose = verbose

        # PPO parameters
        self._gamma = gamma
        self._lmbda = lmbda
        self.epsilon = epsilon
        self.c2 = c2

        # deep learning parameters
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.input_shape = input_shape
        self.n_outputs = n_outputs

        # max timesteps length of an episode
        self.env_max_timesteps = env_max_timesteps

        # builds the Actor and Critic networks (two separate neural networks)
        self._build_network(input_shape, n_outputs, init_personal = init_personal, units=units)

        # actions -> array mask
        self.actions_array = {
            0: np.transpose([1, 0, 0]),
            1: np.transpose([0, 1, 0]),
            2: np.transpose([0, 0, 1]),
        }

        # set up for Tensorboard logging
        if log == True:
            self.log_dir = log_dir
            self.log = log
            self.logger = tf.summary.create_file_writer(log_dir + "/metrics")


    def get_value(self, x):
        return self.critic(x)


    def get_action(self, x, action=None):
        '''Returns the selected action and the probability of that action'''
        logits = self.actor(x)
        probs = tfp.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample(1)
            #prob = probs.prob(action)
        return action.numpy(), probs.log_prob(action).numpy(), probs.entropy()


    def _build_network(self, input_shape, n_outputs, init_personal = True, units=32):

        self._build_policy(input_shape, n_outputs, init_personal = init_personal, units=units)
        self._build_vfunction(input_shape, init_personal = init_personal, units=units)

        self.optimizer_actor = keras.optimizers.Adam(learning_rate=self.lr_actor)
        self.optimizer_critic = keras.optimizers.Adam(learning_rate=self.lr_critic)
        self.optimizer_actor.build(self.actor.trainable_variables)
        self.optimizer_critic.build(self.critic.trainable_variables)


    # actor
    
    def _build_policy(self, input_shape, n_outputs, init_personal = True, units=32):
        
        if init_personal:
            k_initializer_1 = tf.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=2)
            k_initializer_2 = tf.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=1)
            pol_initializer = tf.keras.initializers.Orthogonal(gain=0.01, seed=3)
    
        activ="tanh"

        self.actor = keras.Sequential([
                Dense(units, name = 'actor_dense_1', activation=activ, input_shape=input_shape,
                        kernel_initializer=k_initializer_1 if init_personal else 'glorot_uniform'),
        
                Dense(units, name = 'actor_dense_2', activation=activ, 
                        kernel_initializer=k_initializer_2 if init_personal else 'glorot_uniform'),

                Dense(n_outputs, name='actor_dense_output', activation="linear",
                        kernel_initializer=pol_initializer if init_personal else 'glorot_uniform')
        ])
 

    # critic
    
    def _build_vfunction(self, input_shape, init_personal = True, units=32):
        
        if init_personal:
            k_initializer_1 = tf.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=100)
            k_initializer_2 = tf.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=101)
            val_initializer = tf.keras.initializers.Orthogonal(gain=1, seed=33)
        
        activ='tanh'
        
        self.critic = keras.Sequential([
            Dense(units, name = 'critic_dense_1', activation=activ, input_shape=input_shape,
                            kernel_initializer=k_initializer_1 if init_personal else 'glorot_uniform'),
            
            Dense(units, name = 'critic_dense_2', activation=activ, 
                            kernel_initializer=k_initializer_2 if init_personal else 'glorot_uniform'),
            
            Dense(1, name='critic_dense_output', activation="linear",
                                kernel_initializer=val_initializer if init_personal else 'glorot_uniform')
        ])


    # performs the computation and application of the gradients on the two neural networks
    # the training step is performed on a single minibatch
    
    @tf.function
    def train_step_ppo(self, obs, actions, adv, probs, returns):

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:

            logits = self.actor(obs)
            dist = tfp.distributions.Categorical(logits=logits)
            
            new_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            
            new_value = tf.squeeze(self.critic(obs))
            
            logdif = new_probs - probs
            
            ratio = tf.math.exp(logdif)
            clip = tf.clip_by_value(ratio, 1-self.epsilon_t, 1+self.epsilon_t)# * adv
            
            loss_a_clip = tf.multiply(clip, adv)
            loss_a = tf.multiply(ratio, adv)

            loss_value = tf.reduce_mean((new_value - returns) ** 2)

            loss_actor = tf.negative(tf.reduce_mean(tf.minimum(loss_a_clip, loss_a)))
            
            entropy = tf.reduce_mean(entropy)
            loss_actor = loss_actor - self.c2*entropy

        grads_critic = tape1.gradient(loss_value, self.critic.trainable_variables)
        grads_actor = tape2.gradient(loss_actor, self.actor.trainable_variables)
        self.optimizer_critic.apply_gradients(zip(grads_critic, self.critic.trainable_variables))
        self.optimizer_actor.apply_gradients(zip(grads_actor, self.actor.trainable_variables))

        return loss_actor, loss_value, entropy


    # saves a checkpoint of the models on the disk for future utilization
    
    def save_models(self, addstr = ""):
        v = "" if addstr == "" else "-"
        self.actor.save_weights("models/actor/final" +f"{v}{addstr}" +"/checkpoint.ckpt")
        self.critic.save_weights("models/critic/final" +f"{v}{addstr}" +"/checkpoint.ckpt")


    # loads a checkpoint of the models on the disk and apply the weights to a new neural
    # network with the same architecture
    
    def load_models(self, addstr = ""):
        v = "" if addstr == "" else "-"
        self.actor.load_weights("models/actor/final" +f"{v}{addstr}" +"/checkpoint.ckpt")
        self.critic.load_weights("models/critic/final" +f"{v}{addstr}" +"/checkpoint.ckpt")
        self.optimizer_actor.build(self.actor.trainable_variables)
        self.optimizer_critic.build(self.actor.trainable_variables)

    
    # plays a single timesteps for all the environments in the vector, calculating the 
    # correct action for each of them. The memory is updated with the observation of the
    # new state

    def play_one_step(self, envs: gym.vector.VectorEnv, m: memory.Memory, step, observation):
        
        m.obss[step] = observation
        
        logits = self.actor(observation, training=False)
        actions = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
        
        probs = tf.nn.log_softmax(logits)
        probs = tf.reduce_sum(tf.one_hot(actions, self.n_outputs) * probs, axis=1)
        
        m.actions[step] = actions
        m.probs[step] = probs
        m.values[step] = tf.squeeze(self.get_value(observation))
        
        # make the step(s)
        observation, reward, terminated, truncated, info = envs.step(actions.numpy().squeeze().tolist())

        m.rewards[step] = reward
        m.terminateds[step] = terminated
        m.truncateds[step] = truncated
        
        return observation


    # this is the training loop. It iterates as much as needed, depending on the t_timesteps parameter
    
    def play_n_timesteps(self, envs: gym.vector.VectorEnv, m: memory.Memory, t_timesteps, single_batch_ts, minibatch_size, epochs, difficulty = 0, gamemode='standard'):

        # t_eval_rewards is a vector containing the sums of the reward of every
        # environment inside the context of a batch
        t_eval_rewards = np.zeros(shape=(envs.num_envs,))

        # m_eval_rewards is a variable containing the incremental mean of the
        # cumulative rewards of the finished episode. It get's a reset to 0
        # after each evaluation step of the architecture 
        m_eval_rewards = 0

        # counter used to perform the incremental mean
        e = 0


        # evaluation settings
        # an evaluation step is performed every 50 updates
        eval_frequency = 50
        evaluation = 0 # evaluation counter
        good = 0 # how many consecutive eval score > 0

        # each evaluation step is based on 10 simulated episodes
        eval_n_episodes = 10
        # the episodes used for the evaluation are always generated by the same seeds
        eval_seeds = [int(x) for x in np.random.randint(1, 100000 + 1, size=eval_n_episodes)]


        # prepare the environments for training
        observation, info = envs.reset()

        # batch indexes
        bch_ids = np.arange(0, single_batch_ts, 1)
        
        updates = t_timesteps // single_batch_ts
        
        
        for update in tqdm(range(updates)):

            # memory reset to delete informations about the last simulated batch
            m.reset()

            # incremental means that will be logged using Tensorboard
            mean_actor_loss = 0
            mean_critic_loss = 0
            mean_entropy = 0
            c = 0   # counter

            # update alpha to linearly anneal the learning rates and the epsilon
            alpha = 1 - (update/updates)

            # anneal the optimizer's learning rate
            self.optimizer_actor.learning_rate = self.lr_actor * alpha
            self.optimizer_critic.learning_rate = self.lr_critic * alpha

            # anneal epsilon
            self.epsilon_t = self.epsilon * alpha

            # perform the actions to fill the batch memory with observations, rewards etc.
            for t in range(single_batch_ts):
            
                observation = self.play_one_step(envs, m, t, observation)

                t_eval_rewards += m.rewards[t]

                ids_done = [m.terminateds[t,i] == 1 or m.truncateds[t,i] == 1 for i in range(len(t_eval_rewards))]

                tmp = np.logical_or(m.terminateds[t], m.truncateds[t])
                if np.count_nonzero(tmp) > 0:

                    # indexes of environments that terminated an episode
                    ids_done = np.argwhere(tmp).flatten()

                    # iterate through the finished episode and update the incremental mean of the cumulative sum of rewards
                    for id in ids_done:
                        m_eval_rewards = utils.incremental_mean(m_eval_rewards, t_eval_rewards[id], e)
                        e += 1

                    # reset the reward sum for the finished episodes
                    t_eval_rewards[ids_done] = 0


            # will be used to compute advantages and returns
            next_val = self.get_value(observation).numpy().squeeze()

            # reward normalization in the range [-1,+1] (to comment or change if the environment is not LifeSteps)
            m.rewards = self.reward_transformer(m.rewards, self.env_max_timesteps, difficulty)
            
            # calculating advantages and returns, filling the memory m
            utils.calc_advantages(single_batch_ts, m.advantages, m.rewards, m.values, next_val, self._gamma, self._lmbda, m.terminateds, m.truncateds)
            utils.calc_returns(single_batch_ts, m.returns, m.rewards, next_val, self._gamma, m.terminateds, m.truncateds)

            # flatten all the numpy structures in the memory
            m.flatten()

            # iterate for K epochs over the same batch, shuffling the indexes of the minibatches
            for epoch in range(epochs):

                np.random.shuffle(bch_ids)

                for start_ind in np.arange(0, single_batch_ts, minibatch_size):

                    end_ind = start_ind + minibatch_size
                    ids = bch_ids[start_ind:end_ind]    # minibatch indexes

                    # advantages normalization
                    mb_adv = np.array(m.f_advantages[ids])
                    mb_adv = (mb_adv - np.mean(mb_adv)) / (np.std(mb_adv) + 1e-8)
                    
                    # optimization step
                    loss_actor, loss_critic, entropy = self.train_step_ppo(m.f_obss[ids],
                                                                           m.f_actions[ids],
                                                                           mb_adv,
                                                                           m.f_probs[ids],
                                                                           m.f_returns[ids])

                    # update training metrics
                    mean_actor_loss = utils.incremental_mean(mean_actor_loss, np.mean(loss_actor), c)
                    mean_critic_loss = utils.incremental_mean(mean_critic_loss, np.mean(loss_critic), c)
                    mean_entropy = utils.incremental_mean(mean_entropy, entropy, c)
                    c += 1
                    
            # Tensorboard logging
            if self.log:
                with self.logger.as_default():
                    tf.summary.scalar('loss_actor', data=mean_actor_loss, step=update)
                    tf.summary.scalar('loss_critic', data=mean_critic_loss, step=update)
                    tf.summary.scalar('entropy', data=mean_entropy, step=update)
                    if e > 0:
                        tf.summary.scalar('m_eval_rewards', data=m_eval_rewards, step=update)
                    tf.summary.scalar('lr_critic', data=self.optimizer_critic.learning_rate, step=update)
                    tf.summary.scalar('lr_actor', data=self.optimizer_actor.learning_rate, step=update)
                    tf.summary.scalar('returns mean', data=np.mean(m.f_returns), step=update)
                    tf.summary.scalar('advantages mean', data=np.mean(m.f_advantages), step=update)


            # Evaluation step
            if update % eval_frequency == 0:
                print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
                evaluation += 1
                r = self.evaluate(eval_n_episodes, eval_seeds, evaluation, training = True, log=self.log, render_mode=None, difficulty=difficulty, gamemode=gamemode)
                print()
                
                # resetting the evaluation mean reward to have a realistic value, since the evaluation
                # steps are not very frequent
                m_eval_rewards = 0
                e = 0

                # stop the training process only with 6 consecutive mean score > 0
                if r > 0:
                    good += 1
                    if good == 6:
                        return
                else:
                    good = 0
                
    # reward normalization ONLY TO BE USED WITH LifeSteps CUSTOM ENVIRONMENT
    def reward_transformer(self, rewards, max_ts, difficulty):
        
        return np.where(rewards < 0, rewards/(max_ts+difficulty), rewards)


    # evaluation method
    def evaluate(self, episodes, seeds, eval_code=None, training = False, log = False, render_mode=None, difficulty = 0, gamemode = 'standard'):

        #env = gym.make('CartPole-v1', render_mode='text', max_timesteps=self.env_max_timesteps)
        #env = gym.make('life_sim/LifeSim-v0', render_mode=render_mode, max_timesteps=self.env_max_timesteps)
        env = gym.make('life_steps/LifeSteps-v0', render_mode=render_mode, max_timesteps=self.env_max_timesteps, difficulty = difficulty, gamemode = gamemode)

        mean_cumulative_rewards = 0
        mean_actions = np.zeros(shape=(3,))

        n_episodes = episodes
        
        np.random.shuffle(seeds)

        for episode in np.arange(0, n_episodes, 1):

            sum_rewards = 0
            actions = np.transpose([0, 0, 0])

            obs, info = env.reset(seed=seeds[episode])
            
            while True:    

                action, _ , _= self.get_action(obs[np.newaxis])

                action = action.squeeze().tolist()
                obs, reward, terminated, truncated, info = env.step(action)
            
                actions = actions + self.actions_array[action]

                sum_rewards += reward

                if render_mode != 'human' and render_mode != None:
                    env.render()
                if terminated or truncated:
                    break

            # normalize actions selection to get percentages of selected actions
            m_a = np.sum(actions)
            actions = actions / m_a
            mean_actions = [utils.incremental_mean(mean_actions[id], actions[id], episode) for id in range(len(actions))]

            mean_cumulative_rewards = utils.incremental_mean(mean_cumulative_rewards, sum_rewards, episode)

        # Tensorboard logging
        if log:
            if training == True:
                with self.logger.as_default():
                    tf.summary.scalar('cumulative_rewards', data=mean_cumulative_rewards, step=eval_code, description="Metric inside training")
                    tf.summary.scalar('chosen_work', data=mean_actions[0], step=eval_code, description="Metric inside training")
                    tf.summary.scalar('chosen_sport', data=mean_actions[1], step=eval_code, description="Metric inside training")
                    tf.summary.scalar('chosen_sociality', data=mean_actions[2], step=eval_code, description="Metric inside training")
        
        print(">>> Evaluation <<<")
        print(f"\tMean Cumulative Rewards: {mean_cumulative_rewards:.3f}")
        
        env.close()

        if not log:
            return mean_cumulative_rewards, mean_actions
        
        return mean_cumulative_rewards