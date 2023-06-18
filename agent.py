'''this file provides the Agent abstract class and some of its useful implementations'''
from abc import ABC
import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow_probability as tfp
import gymnasium as gym
from tqdm.notebook import tqdm
import memory
import utils
import datetime
import os


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
    def __init__(self, input_shape, n_outputs, gamma, lmbda, epsilon, c2, lr_actor, lr_critic, log_dir, verbose=0):

        self._verbose = verbose

        # self.m = memory
        self._gamma = gamma
        self._lmbda = lmbda
        self.epsilon = epsilon
        self.c2 = c2

#        self._optimizer_actor = keras.optimizers.AdamW(learning_rate=lr)
#        self._optimizer_critic = keras.optimizers.AdamW(learning_rate=lr)
        # self._actor_loss_fn = actor_loss_fn
        # self._critic_loss_fn = critic_loss_fn

        # self._discount = discount_factor

        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.input_shape = input_shape
        self.n_outputs = n_outputs

        self._build_network(input_shape, n_outputs)
    
        #self.log_dir = log_dir
        self.file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
        self.file_writer.set_as_default()

        #self._act_array = tf.constant([0, 1, 2])

        self.optimizer_actor.build(self.actor.trainable_variables)
        self.optimizer_critic.build(self.critic.trainable_variables)

    def get_value(self, x):
        return self.critic(x)

    def get_action(self, x, action=None):
        '''Returns the selected action and the probability of that action'''
        logits = self.actor(x)
        probs = tfp.distributions.Categorical(probs=logits)
        if action is None:
            action = probs.sample(1)
            prob = probs.prob(action)
        else:
            prob = probs.prob(action)
        return action, prob, probs.entropy()

    def get_logit(self, logits, actions):
        ll = []
        for a in range(len(actions)):
            ll.append(np.array([logits[a][aa] for aa in actions[a]]))
        return ll

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = tfp.distributions.Categorical(probs=logits)
        if action is None:
            action = probs.sample(1)
        return action, probs.prob(action), probs.entropy(), self.critic(x)

    def play_one_step(self, env, state, epsilon):

        action = np.argmax(self._policy(state, epsilon))
        next_state, reward, terminated, truncated, info = env.step(action)
        return next_state, reward, terminated, truncated, info, action

    def _build_network(self, input_shape, n_outputs):
        #self.optimizer_actor = keras.optimizers.AdamW(learning_rate=self.lr_actor)
        #self.optimizer_critic = keras.optimizers.AdamW(learning_rate=self.lr_critic)
        self.optimizer_actor = keras.optimizers.AdamW(learning_rate=self.lr_actor)
        self.optimizer_critic = keras.optimizers.AdamW(learning_rate=self.lr_critic)
        self._build_policy(input_shape, n_outputs)
        self._build_vfunction(input_shape)

    # actor
    def _build_policy(self, input_shape, n_outputs):
        bias_init = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=42)
        self.actor = keras.Sequential([
            keras.layers.Dense(128, name='actor_dense_1', activation="relu", input_shape=input_shape,
                               kernel_initializer='random_uniform', bias_initializer=bias_init),
            keras.layers.Dense(128, name='actor_dense_2', activation="relu", kernel_initializer='random_uniform',
                               bias_initializer=bias_init),
            keras.layers.Dense(units=n_outputs, name='actor_dense_output', activation="relu", kernel_initializer='random_uniform',
                               bias_initializer=bias_init),
            keras.layers.Softmax(name='actor_dense_softmax')
        ])

    # critic
    def _build_vfunction(self, input_shape):
        bias_init = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=19)
        self.critic = keras.Sequential([
            keras.layers.Dense(128, name='critic_dense_1', activation="relu", input_shape=input_shape,
                               kernel_initializer='random_uniform',
                               bias_initializer=bias_init),
            keras.layers.Dense(128, name='critic_dense_2', activation="relu", input_shape=input_shape,
                               kernel_initializer='random_uniform',
                               bias_initializer=bias_init),
            keras.layers.Dense(1, name='critic_dense_output')
        ])

    @tf.function
    def train_step_ppo(self, obs, actions, adv, probs, returns):
        
        #with tf.device('/physical_device/CPU:0'):
        with tf.GradientTape(persistent=True) as tape:
                
            _, new_probs, entropy = self.get_action(obs, actions)
            new_value = self.get_value(obs)
            #self.get_action_and_value
            #new_probs = get_prob_from_action(new_probs, mb_action)

            ratio = tf.divide(new_probs, probs)
            
            clip = tf.clip_by_value(ratio, 1-self.epsilon, 1+self.epsilon) * adv
            #clip = np.clip(ratio, 1-self.epsilon, 1+self.epsilon) * adv
            loss_clip = tf.minimum(ratio * adv, clip)
            #loss_clip = -tf.reduce_mean(loss_clip)
            loss_clip = tf.reduce_mean(loss_clip)
            
            loss_actor = -loss_clip - self.c2*entropy
            loss_value = tf.reduce_mean(tf.square(returns - new_value))
            #loss_value = tf.keras.losses.mse(m.f_returns[ids], new_value)

            #loss = - loss_clip - loss_value

        grads_actor = tape.gradient(loss_actor, self.actor.trainable_variables)
        grads_critic = tape.gradient(loss_value, self.critic.trainable_variables)
        self.optimizer_actor.apply_gradients(zip(grads_actor, self.actor.trainable_variables))
        self.optimizer_critic.apply_gradients(zip(grads_critic, self.critic.trainable_variables))
            
        return loss_actor, loss_value

    def play_n_timesteps(self, envs: gym.vector.VectorEnv, mem: memory.Memory, t_timesteps, single_batch_ts, minibatch_size, epochs):

        '''
        checkpoint_path_actor = "training_1_actor/model.ckpt"
        checkpoint_path_critic = "training_1_critic/model.ckpt"
        checkpoint_dir_actor = os.path.dirname(checkpoint_path_actor)
        checkpoint_dir_critic = os.path.dirname(checkpoint_path_critic)
        '''

        # Save the weights
        #model.save_weights('./checkpoints/my_checkpoint')

        # Create a new model instance
        #model = create_model()

        # Restore the weights
        #model.load_weights('./checkpoints/my_checkpoint')

        #update_mean_actor_loss = []
        #update_mean_critic_loss = []

        batch = envs.num_envs * single_batch_ts
        updates = t_timesteps // single_batch_ts
        m = mem
        #tau_loss = 
        observation, info = envs.reset()
        
        alpha = 1
        
        for update in tqdm(range(updates)):

            #alpha = 1 - (update/(updates*0.85)) if alpha > 0 else 0
            alpha = 1 - (update/updates)
            lr_actor = self.lr_actor * alpha
            lr_critic = self.lr_critic * alpha
            # create new adam (weight decay = 0, lr updated)
            ###self.optimizer_actor = keras.optimizers.Adam(learning_rate=self.lr_critic)
            self.optimizer_actor.learning_rate = lr_actor
            ###self.optimizer_critic = keras.optimizers.Adam(learning_rate=self.lr_critic)
            self.optimizer_critic.learning_rate = lr_critic

            #print(f"a: {alpha}")
            #print(f"e: {self.epsilon}")
            self.epsilon = self.epsilon * alpha

            for t in range(single_batch_ts):
            
                m.obss[t] = observation

                actions, probs, _ = self.get_action(observation)
                m.values[t] = self.get_value(observation).numpy().squeeze()

                # the Gymnasium Vectorized Environment step method takes a 1-dimensional list of actions as parameter  
                observation, reward, terminated, truncated, info = envs.step(actions.numpy().squeeze().tolist())
                
                m.actions[t] = actions
                m.probs[t] = probs
                m.rewards[t] = reward
                m.terminateds[t] = terminated
                m.truncateds[t] = truncated
                m.values[t] 

            m.obss[t] = observation

            #next_val = self.get_value(observation[np.newaxis]).numpy().squeeze()
            next_val = self.get_value(observation).numpy().squeeze()

            # calc advantages
            m.advantages = np.array([utils.calc_adv_list(single_batch_ts-1, 0, m.rewards[:, env_id], m.values[:, env_id], self._gamma,
                                                self._lmbda, m.terminateds[:, env_id], m.truncateds[:, env_id]) for env_id in range(m.num_envs)], dtype=np.float32)
            # calc returns
            m.returns = np.array([utils.calc_returns(single_batch_ts-1, 0, m.rewards[:, env_id], self._gamma,
                                            m.terminateds[:, env_id], m.truncateds[:, env_id]) for env_id in range(m.num_envs)], dtype=np.float32)

            m.flatten()

            mb_actor_loss_l = []
            mb_critic_loss_l = []

            for epoch in range(epochs):

                start_ind = epoch * minibatch_size
                end_ind = start_ind + minibatch_size

                ids = np.arange(start_ind, end_ind, 1)

                for mb in range(minibatch_size):

                    mb_obs = np.array(m.f_obss[ids])
                    mb_actions = np.array(m.f_actions[ids])
                    mb_adv = np.array(m.f_advantages[ids])
                    mb_probs = np.array(m.f_probs[ids])
                    mb_returns = np.array(m.f_returns[ids])
                    
                    loss_actor, loss_value = self.train_step_ppo(mb_obs, mb_actions, mb_adv, mb_probs, mb_returns)

                    mb_actor_loss_l.append(loss_actor)
                    mb_critic_loss_l.append(loss_value)

            #update_mean_actor_loss.append(np.mean(mb_actor_loss_l))
            #update_mean_critic_loss.append(np.mean(mb_critic_loss_l))
        

#            with self.tf_writer.as_default():
            tf.summary.scalar('loss_actor', data=np.mean(mb_actor_loss_l), step=update)
            tf.summary.scalar('loss_critic', data=np.mean(mb_critic_loss_l), step=update)


            #self.actor.save_weights(checkpoint_dir_actor)
            #self.critic.save_weights(checkpoint_dir_critic)

        return [], []
    
    def save_models(self, addstr = ""):
        v = "" if addstr == "" else "-"
        self.actor.save(filepath="models/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/actor" +f"{v}{addstr}" + "/")
        self.critic.save(filepath="models/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/critic" +f"{v}{addstr}" + "/")

#    def evaluate(self, envs :gym.vector.VectorEnv,)