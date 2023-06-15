'''this file provides the Agent abstract class and some of its useful implementations'''
from abc import ABC
import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import gymnasium as gym
from tqdm.notebook import tqdm
import memory
import utils


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


class DQN_Agent(Agent):

    def __init__(self, input_shape, n_outputs, optimizer, loss_fn, discount_factor):

        self._optimizer = optimizer
        self._loss_fn = loss_fn
        self._discount = discount_factor

        self.input_shape = input_shape
        self.n_outputs = n_outputs

        self.dqn_model = keras.Sequential([
            keras.layers.Dense(64, activation="relu", input_shape=input_shape,
                               kernel_initializer='random_uniform',
                               bias_initializer=keras.initializers.Constant(0.1)),
            keras.layers.Dense(n_outputs)
        ])

    # implementation of an Epsilon Greedy Policy
    def epsilon_greedy_policy(self, state, epsilon=0):

        if np.random.rand() < epsilon:
            return np.random.randint(3)
        else:
            q_values = self.dqn_model.predict(state[np.newaxis], verbose=0)
            return np.argmax(q_values)

    def _policy(self, state, epsilon):

        return self.epsilon_greedy_policy(state, epsilon)

    def predict(self, state, epsilon):
        return int(self._policy(state, epsilon))

    def play_one_step(self, env, state, epsilon):

        action = self.predict(state, epsilon)
        next_state, reward, terminated, truncated, info = env.step(action)
        return next_state, reward, terminated, truncated, info, action

    def _sample_experiences(self, replay_buffer, batch_size):
        indices = np.random.randint(len(replay_buffer), size=batch_size)
        batch = [replay_buffer[index] for index in indices]
        states, actions, rewards, next_states, terminateds = [np.array([experience[field_index] for experience in batch])
                                                              for field_index in range(5)]
        return states, actions, rewards, next_states, terminateds

    def train(self, replay_buffer, batch_size):

        batch = self._sample_experiences(replay_buffer, batch_size)
        states, actions, rewards, next_states, terminateds = batch

        next_q_values = self.dqn_model.predict(next_states, verbose=0)
        max_next_q_values = np.max(next_q_values, axis=1)
        target_q_values = (rewards + (1-terminateds) *
                           self._discount*max_next_q_values)

        mask = tf.one_hot(actions, self.n_outputs)
        with tf.GradientTape() as tape:

            all_q_values = self.dqn_model(states)
            q_values = tf.reduce_sum(all_q_values*mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self._loss_fn(target_q_values, q_values))

        grads = tape.gradient(loss, self.dqn_model.trainable_variables)
        self._optimizer.apply_gradients(
            zip(grads, self.dqn_model.trainable_variables))
        return loss


class PPO_Agent(Agent):
    def __init__(self, input_shape, n_outputs, gamma, lmbda, epsilon, lr, verbose=0):

        '''physical_devices = tf.config.list_physical_devices('GPU')
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
        # Invalid device or cannot modify virtual devices once initialized.
            pass
        '''
        self._verbose = verbose

        # self.m = memory
        self._gamma = gamma
        self._lmbda = lmbda
        self.epsilon = epsilon

#        self._optimizer_actor = keras.optimizers.AdamW(learning_rate=lr)
#        self._optimizer_critic = keras.optimizers.AdamW(learning_rate=lr)
        # self._actor_loss_fn = actor_loss_fn
        # self._critic_loss_fn = critic_loss_fn

        # self._discount = discount_factor

        self.lr = lr
        self.input_shape = input_shape
        self.n_outputs = n_outputs

        self._build_network(input_shape, n_outputs)
        
        
        #self._act_array = tf.constant([0, 1, 2])

        #self._optimizer_actor.build(self.actor.trainable_variables)
        #self._optimizer_critic.build(self.critic.trainable_variables)

    def get_value(self, x):
        return self.critic(x)

    def get_action(self, x, action=None):
        '''Returns the selected action and the probability of that action'''
        logits = self.actor(x)
        probs = tfd.Categorical(probs=logits)
        if action is None:
            action = probs.sample(1)
            prob = probs.prob(action)
        else:
            prob = probs.prob(action)
        return action, prob

    def get_logit(self, logits, actions):
        ll = []
        for a in range(len(actions)):
            ll.append(np.array([logits[a][aa] for aa in actions[a]]))
        return ll

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = tfd.Categorical(probs=logits)
        if action is None:
            action = probs.sample(1)
        return action, probs.prob(action), self.critic(x)

    def play_one_step(self, env, state, epsilon):

        action = np.argmax(self._policy(state, epsilon))
        next_state, reward, terminated, truncated, info = env.step(action)
        return next_state, reward, terminated, truncated, info, action

    def _build_network(self, input_shape, n_outputs):
        self.optimizer_actor = keras.optimizers.AdamW(learning_rate=self.lr)
        self.optimizer_critic = keras.optimizers.AdamW(learning_rate=self.lr)
        self._build_policy(input_shape, n_outputs)
        self._build_vfunction(input_shape)

    # actor
    def _build_policy(self, input_shape, n_outputs):
        self.actor = keras.Sequential([
            keras.layers.Dense(8, name='actor_dense_1', activation="relu", input_shape=input_shape,
                               kernel_initializer='random_uniform', bias_initializer=keras.initializers.Constant(0.1)),
            keras.layers.Dense(units=n_outputs, name='actor_dense_2', activation="relu", kernel_initializer='random_uniform',
                               bias_initializer=keras.initializers.Constant(0.1)),
            keras.layers.Softmax(name='actor_dense_softmax')
        ])

    # critic
    def _build_vfunction(self, input_shape):
        self.critic = keras.Sequential([
            keras.layers.Dense(4, name='critic_dense_1', activation="relu", input_shape=input_shape,
                               kernel_initializer='random_uniform',
                               bias_initializer=keras.initializers.Constant(0.1)),
            keras.layers.Dense(1, name='critic_dense_2')
        ])

    @tf.function
    def train_step_ppo(self, obs, adv, probs, returns):
        
        #with tf.device('/physical_device/CPU:0'):
        with tf.GradientTape(persistent=True) as tape:
                
            _, new_probs = self.get_action(obs)
            new_value = self.get_value(obs)
            #self.get_action_and_value
            #new_probs = get_prob_from_action(new_probs, mb_action)

            ratio = tf.divide(new_probs, probs)
            
            clip = tf.clip_by_value(ratio, 1-self.epsilon, 1+self.epsilon) * adv
            #clip = np.clip(ratio, 1-self.epsilon, 1+self.epsilon) * adv
            loss_clip = tf.minimum(ratio * adv, clip)
            loss_clip = -tf.reduce_mean(loss_clip)
            #loss_clip = tf.reduce_mean(loss_clip)
            
            loss_value = tf.reduce_mean(tf.square(returns - new_value))
            #loss_value = tf.keras.losses.mse(m.f_returns[ids], new_value)

            #loss = - loss_clip - loss_value

        grads_actor = tape.gradient(loss_clip, self.actor.trainable_variables)
        grads_critic = tape.gradient(loss_value, self.critic.trainable_variables)
        self.optimizer_actor.apply_gradients(zip(grads_actor, self.actor.trainable_variables))
        self.optimizer_critic.apply_gradients(zip(grads_critic, self.critic.trainable_variables))
            
        return loss_clip, loss_value

    def play_n_timesteps(self, envs: gym.vector.VectorEnv, mem: memory.Memory, t_timesteps, single_batch_ts, minibatch_size, epochs):

#        self._build_network(a.input_shape, a.n_outputs)

        batch = envs.num_envs * single_batch_ts
        updates = t_timesteps // single_batch_ts
        m = mem

        observation, info = envs.reset()
        
        for update in tqdm(range(updates)):
            #print(update)
            for t in range(single_batch_ts):
                #print(t)
                m.obss[t] = observation

                actions, probs = self.get_action(observation)
                observation, reward, terminated, truncated, info = envs.step(actions.numpy().squeeze().tolist())

                m.rewards[t] = reward
                m.terminateds[t] = terminated
                m.truncateds[t] = truncated

            m.obss[t] = observation

            next_val = self.get_value(observation[np.newaxis]).numpy().squeeze()

            # calc advantages
            m.advantages = np.array([utils.calc_adv_list(single_batch_ts-1, 0, m.rewards[:, env_id], m.values[:, env_id], self._gamma,
                                                self._lmbda, m.terminateds[:, env_id], m.truncateds[:, env_id], next_val[env_id]) for env_id in range(m.num_envs)], dtype=np.float32)
            # calc returns
            m.returns = np.array([utils.calc_returns(single_batch_ts-1, 0, m.rewards[:, env_id], self._gamma,
                                            m.terminateds[:, env_id], m.truncateds[:, env_id]) for env_id in range(m.num_envs)], dtype=np.float32)

            m.flatten()

            for epoch in range(epochs):

                start_ind = epoch * minibatch_size
                end_ind = start_ind + minibatch_size

                ids = np.arange(start_ind, end_ind, 1)

                for mb in range(minibatch_size):

                    mb_obs = np.array(m.f_obss[ids])
                    mb_adv = np.array(m.f_advantages[ids])
                    mb_probs = np.array(m.f_probs[ids])
                    mb_returns = np.array(m.f_returns[ids])
                    
                    loss_actor, loss_value = self.train_step_ppo(mb_obs, mb_adv, mb_probs, mb_returns)

    '''
    def play_n_timesteps(self, envs: gym.vector.VectorEnv, mem: memory.Memory, t_timesteps, single_batch_ts, minibatch_size, epochs):

        batch = envs.num_envs * single_batch_ts
        updates = t_timesteps // single_batch_ts
        m = mem

        observation, info = envs.reset()
        
        for update in tqdm(range(updates)):
            print(update)
            for t in range(single_batch_ts):
                print(t)
                m.obss[t] = observation

                actions, probs = self.get_action(observation)
                observation, reward, terminated, truncated, info = envs.step(actions)

                m.rewards[t] = reward
                m.terminateds[t] = terminated
                m.truncateds[t] = truncated

            m.obss[t] = observation

            next_val = self.get_value(observation)

            # calc advantages
            m.advantages = [utils.calc_adv_list(single_batch_ts-1, 0, m.rewards[:, env_id], m.values[:, env_id], self._gamma,
                                                self._lmbda, m.terminateds[:, env_id], m.truncateds[:, env_id], next_val[env_id]) for env_id in range(m.num_envs)]
            # calc returns
            m.returns = [utils.calc_returns(single_batch_ts-1, 0, m.rewards[:, env_id], self._gamma,
                                            m.terminateds[:, env_id], m.truncateds[:, env_id]) for env_id in range(m.num_envs)]

            m.flatten()

            for epoch in range(epochs):

                start_ind = epoch * minibatch_size
                end_ind = start_ind + minibatch_size

                ids = np.arange(start_ind, end_ind, 1)

                for mb in range(minibatch_size):

                    mb_obs = np.array(m.f_obss[ids])
                    mb_adv = np.array(m.f_advantages[ids])
                    mb_probs = np.array(m.f_probs[ids])
                    mb_returns = np.array(m.f_returns[ids])
                    
                    loss_actor, loss_value = self.train_step_ppo(mb_obs, mb_adv, mb_probs, mb_returns)
                    #loss_actor, loss_value = self.train_step_ppo(m, ids)
                    
                    with tf.GradientTape() as tape_actor, tf.GradientTape() as tape_critic:

                        _, values = self.get_action(
                            m.f_obs[ids], m.f_actions[ids])
                    
                        ratio = tf.divide(values, m.f_probs[ids])
                        obj_clip = tf.minimum(ratio*m.f_advantages[ids], tf.clip_by_value(
                            ratio, 1-self.epsilon, 1+self.epsilon)*m.f_advantages[ids])
                        obj_clip = tf.reduce_mean(obj_clip)
                        loss_clip = tf.negative(obj_clip)
                        loss_value = tf.losses.MSE(values, m.f_returns[ids])
                    
                    grads_actor
                    


@tf.function
def train_step_ppo(a, obs, adv, probs, returns):
    
    #with tf.device('/physical_device/CPU:0'):
    with tf.GradientTape(persistent=True) as tape:
            
        _, new_probs = a.get_action(obs)
        new_value = a.get_value(obs)

        #new_probs = get_prob_from_action(new_probs, mb_action)

        ratio = np.divide(new_probs, probs)
        
        #clip = tf.clip_by_value(ratio, 1-self.epsilon, 1+self.epsilon) * m.f_advantages[ids]
        clip = np.clip(ratio, 1-a.epsilon, 1+a.epsilon) * adv
        loss_clip = tf.minimum(ratio * adv, clip)
        loss_clip = -tf.reduce_mean(loss_clip)
        #loss_clip = tf.reduce_mean(loss_clip)
        
        loss_value = tf.reduce_mean(tf.square(returns - new_value))
        #loss_value = tf.keras.losses.mse(m.f_returns[ids], new_value)

        #loss = - loss_clip - loss_value

    grads_actor = tape.gradient(loss_clip, a.actor.trainable_variables)
    grads_critic = tape.gradient(loss_value, a.critic.trainable_variables)
    a.optimizer_actor.apply_gradients(zip(grads_actor, a.actor.trainable_variables))
    a.optimizer_critic.apply_gradients(zip(grads_critic, a.critic.trainable_variables))
        
    return loss_clip, loss_value

def play_n_timesteps(a, envs: gym.vector.VectorEnv, mem: memory.Memory, t_timesteps, single_batch_ts, minibatch_size, epochs):

    a._build_network(a.input_shape, a.n_outputs)

    batch = envs.num_envs * single_batch_ts
    updates = t_timesteps // single_batch_ts
    m = mem

    observation, info = envs.reset()
    
    for update in tqdm(range(updates)):
        print(update)
        for t in range(single_batch_ts):
            #print(t)
            m.obss[t] = observation

            actions, probs = a.get_action(observation)
            observation, reward, terminated, truncated, info = envs.step(actions.numpy().squeeze().tolist())

            m.rewards[t] = reward
            m.terminateds[t] = terminated
            m.truncateds[t] = truncated

        m.obss[t] = observation

        next_val = a.get_value(observation[np.newaxis]).numpy().squeeze()

        # calc advantages
        m.advantages = np.array([utils.calc_adv_list(single_batch_ts-1, 0, m.rewards[:, env_id], m.values[:, env_id], a._gamma,
                                            a._lmbda, m.terminateds[:, env_id], m.truncateds[:, env_id], next_val[env_id]) for env_id in range(m.num_envs)], dtype=np.float32)
        # calc returns
        m.returns = np.array([utils.calc_returns(single_batch_ts-1, 0, m.rewards[:, env_id], a._gamma,
                                        m.terminateds[:, env_id], m.truncateds[:, env_id]) for env_id in range(m.num_envs)], dtype=np.float32)

        m.flatten()

        for epoch in range(epochs):

            start_ind = epoch * minibatch_size
            end_ind = start_ind + minibatch_size

            ids = np.arange(start_ind, end_ind, 1)

            for mb in range(minibatch_size):

                mb_obs = np.array(m.f_obss[ids])
                mb_adv = np.array(m.f_advantages[ids])
                mb_probs = np.array(m.f_probs[ids])
                mb_returns = np.array(m.f_returns[ids])
                
                loss_actor, loss_value = a.train_step_ppo(mb_obs, mb_adv, mb_probs, mb_returns)

             
def main(T, actor, critic, env):

    obs, info = env.reset()
    v_reward = []
    v_values = []
    v_terminated = []
    v_truncated = []

    gamma = 1
    lmbda = 1

    for t in T:
        v_values.append(critic.predict(obs))
        action = actor.predict(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        v_reward.append(reward)
        v_terminated.append(terminated)
        v_truncated.append(truncated)

    v_values.append(critic.predict(obs))

    v_values, v_terminated, v_truncated = prep_adv_calc(
        v_values, critic.predict(obs), v_terminated, v_truncated)
    advantages = calc_adv(T, t, v_reward, v_values, gamma,
                          lmbda, v_terminated, v_truncated)
'''