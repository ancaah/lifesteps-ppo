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
    def __init__(self, input_shape, n_outputs, gamma, lmbda, epsilon, c2, lr_actor, lr_critic, log = False, log_dir = None, env_max_timesteps=300, verbose=0):

        self._verbose = verbose

        # self.m = memory
        self._gamma = gamma
        self._lmbda = lmbda
        self.epsilon = epsilon
        self.c2 = c2

        self.actions_array = {
            0: np.transpose([1, 0, 0]),
            1: np.transpose([0, 1, 0]),
            2: np.transpose([0, 0, 1]),
        }

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
        if log == True:
            self.log_dir = log_dir
            self.log = log
            self.train_writer = tf.summary.create_file_writer(log_dir + "/metrics/train")
            self.eval_writer = tf.summary.create_file_writer(log_dir + "/metrics/eval")
            #self.file_writer.set_as_default()

        #self._act_array = tf.constant([0, 1, 2])

        self.optimizer_actor.build(self.actor.trainable_variables)
        self.optimizer_critic.build(self.critic.trainable_variables)

        self.env_max_timesteps = env_max_timesteps

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
        self.optimizer_actor = keras.optimizers.Adam(learning_rate=self.lr_actor)
        self.optimizer_critic = keras.optimizers.Adam(learning_rate=self.lr_critic)
        self._build_policy(input_shape, n_outputs)
        self._build_vfunction(input_shape)

    # actor
    def _build_policy(self, input_shape, n_outputs):
        #bias_init = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=42)
        k_initializer = tf.keras.initializers.Orthogonal()
        self.actor = keras.Sequential([
            keras.layers.Dense(128, name='actor_dense_1', activation="relu", input_shape=input_shape,
                               kernel_initializer=k_initializer),
            keras.layers.Dense(128, name='actor_dense_2', activation="relu", kernel_initializer=k_initializer),
            keras.layers.Dense(units=n_outputs, name='actor_dense_output', activation="relu", kernel_initializer=k_initializer),
            keras.layers.Softmax(name='actor_dense_softmax')
        ])

    # critic
    def _build_vfunction(self, input_shape):
#        bias_init = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=19)
        k_initializer = tf.keras.initializers.Orthogonal()
        self.critic = keras.Sequential([
            keras.layers.Dense(256, name='critic_dense_1', activation="relu", input_shape=input_shape,
                               kernel_initializer=k_initializer),
            keras.layers.Dense(128, name='critic_dense_2', activation="relu", input_shape=input_shape,
                               kernel_initializer=k_initializer),
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

    def save_models(self, addstr = ""):
        v = "" if addstr == "" else "-"
        self.actor.save(filepath="models/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/actor" +f"{v}{addstr}" + "/")
        self.critic.save(filepath="models/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/critic" +f"{v}{addstr}" + "/")

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

        #if not self.log:
        update_mean_actor_loss = []
        update_mean_critic_loss = []

        mean_actor_loss = 0
        n_a = 0
        mean_critic_loss = 0
        n_c = 0
        last_mcr = 0
        
        batch = envs.num_envs * single_batch_ts
        updates = t_timesteps // single_batch_ts
        m = mem
        #tau_loss = 
        observation, info = envs.reset()
        
        alpha = 1

        eval_frequency = 2
        evaluation = 0 # evaluation counter

        # seeds for evaluation
        eval_n_episodes = 25
        eval_seeds = [int(x) for x in np.random.randint(1, 100000 + 1, size=eval_n_episodes)]
        eval_env = gym.make('life_sim/LifeSim-v0', render_mode='text', max_timesteps=self.env_max_timesteps)
        
        for update in tqdm(range(updates)):

            mean_cumulative_rewards = 0

            n_r = 0

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

            #next_truncated = np.zeros(shape=(m.truncateds[0]))
            #next_terminated = np.zeros(shape=(m.terminateds[0]))
            
            for t in range(single_batch_ts):
            
                m.obss[t] = observation
                #m.terminateds[t] = next_terminated
                #m.truncateds[t] = next_truncated
                #print(observation)
                actions, probs, _ = self.get_action(observation)
                m.values[t] = self.get_value(observation).numpy().squeeze()

                # the Gymnasium Vectorized Environment step method takes a 1-dimensional list of actions as parameter  
                observation, reward, terminated, truncated, info = envs.step(actions.numpy().squeeze().tolist())
                
                m.actions[t] = actions
                m.probs[t] = probs
                m.rewards[t] = reward
                m.terminateds[t] = terminated
                m.truncateds[t] = truncated
                
                mean_cumulative_rewards = utils.incremental_mean(mean_cumulative_rewards, np.mean(m.rewards[t]), n_r)
                n_r += 1

            #print(f"last {last_mcr}")
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

            bch_ids = np.arange(0, single_batch_ts, 1)

            for epoch in range(epochs):

                np.random.shuffle(bch_ids)

                for mb_start in np.arange(0, single_batch_ts, minibatch_size):

                    start_ind = mb_start
                    end_ind = start_ind + minibatch_size

                    ids = np.arange(start_ind, end_ind, 1)

#                for mb in range(minibatch_size):

                    mb_obs = np.array(m.f_obss[ids])
                    mb_actions = np.array(m.f_actions[ids])
                    mb_adv = np.array(m.f_advantages[ids])
                    mb_probs = np.array(m.f_probs[ids])
                    mb_returns = np.array(m.f_returns[ids])
                    
                    loss_actor, loss_critic = self.train_step_ppo(mb_obs, mb_actions, mb_adv, mb_probs, mb_returns)

                    mean_actor_loss = utils.incremental_mean(mean_actor_loss, np.mean(loss_actor), n_a)
                    mean_critic_loss = utils.incremental_mean(mean_critic_loss, np.mean(loss_critic), n_c)
                    n_a += 1
                    n_c += 1
                    mb_actor_loss_l.append(loss_actor)
                    mb_critic_loss_l.append(loss_critic)

            #m.obss[t] = observation
            #print(mean_cumulative_rewards)
            if update % eval_frequency == 0:
                print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
                evaluation += 1
                self.evaluate(eval_n_episodes, eval_seeds, evaluation)
                #print(f"Mean cumulative rewards of the last batch (per step): {mean_cumulative_rewards:.5}")
                #print(f"Cumulative reward increment (wrt last mean): {mean_cumulative_rewards-last_mcr:.5}")
                print()
                last_mcr = mean_cumulative_rewards

            
            if self.log:
                with self.train_writer.as_default():
                    loss_act = np.mean(mb_actor_loss_l)
                    loss_cri = np.mean(mb_critic_loss_l)
                    tf.summary.scalar('loss_actor', data=loss_act, step=update)
                    tf.summary.scalar('loss_critic', data=loss_cri, step=update)
                    
                    if update % eval_frequency == 0:
                        print(f"Actor loss: {loss_act:.5}")
                        print(f"Critic loss: {loss_cri:.5}")
                        print()
            else:
                update_mean_actor_loss.append(np.mean(mb_actor_loss_l))
                update_mean_critic_loss.append(np.mean(mb_critic_loss_l))
        
        print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
        print("Training results")
        print()
        print(f"Mean Actor Loss: {mean_actor_loss}")
        print(f"Mean Critic Loss: {mean_critic_loss}")
        print(f"Mean cumulative rewards of the last batch (per step): {mean_cumulative_rewards}")
        #print(f"Cumulative reward increment (wrt last mean): {mean_cumulative_rewards-last_mcr}")
        print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
        
        if not self.log:
            return update_mean_actor_loss, update_mean_critic_loss

    def evaluate(self, episodes, seeds, eval_code=None):

        # max number of steps for each episode of the simulator.
        # if reached, the environment is TRUNCATED by the TimeLimit wrapper
        #max_steps = 300

        env = gym.make('life_sim/LifeSim-v0', render_mode='text', max_timesteps=self.env_max_timesteps)

        if not self.log:
            cumulative_rewards = []
            sum_actions = []

        mean_cumulative_rewards = 0
        mean_actions = np.zeros(shape=(3,))

        avg_sum_reward = 0
        avg_sum_actions = np.transpose([0, 0, 0])

        n_episodes = episodes
        t = 0

        #for episode in tqdm(np.arange(1, n_episodes+1, 1), desc="Episodes", position=0):
        for episode in np.arange(0, n_episodes, 1):

            #print(f"Episode {episode}")
            sum_rewards = 0
            actions = np.transpose([0, 0, 0])

            obs, info = env.reset(seed=seeds[t])
            t += 1

            #for step in np.arange(1, max_steps, 1):
            while True:    

                action, _ , _= self.get_action(obs[np.newaxis])

                action = action.numpy().squeeze().tolist()
                new_obs, reward, terminated, truncated, info = env.step(action)

                actions = actions + self.actions_array[action]

                sum_rewards += reward
        #        actions = actions + actions_array[v_info['last_action']]

                #env.render()
                if terminated or truncated:
                    break

            # normalize actions selection
            m_a = np.max(actions)
            actions = np.divide(actions, m_a)
            mean_actions = [utils.incremental_mean(mean_actions[id], actions[id], episode) for id in range(3)]

            mean_cumulative_rewards = utils.incremental_mean(mean_cumulative_rewards, sum_rewards, episode)


        if self.log:
            if eval_code is not None:
                with self.train_writer.as_default():
                    tf.summary.scalar('cumulative_rewards', data=mean_cumulative_rewards, step=eval_code)
                    tf.summary.scalar('chosen_work', data=mean_actions[0], step=eval_code)
                    tf.summary.scalar('chosen_sport', data=mean_actions[1], step=eval_code)
                    tf.summary.scalar('chosen_sociality', data=mean_actions[2], step=eval_code)
            else:
                with self.eval_writer.as_default():
                    tf.summary.scalar('cumulative_rewards', data=mean_cumulative_rewards, step=eval_code)
                    tf.summary.scalar('chosen_work', data=mean_actions[0], step=eval_code)
                    tf.summary.scalar('chosen_sport', data=mean_actions[1], step=eval_code)
                    tf.summary.scalar('chosen_sociality', data=mean_actions[2], step=eval_code)
        else:
            sum_actions.append(actions)
            cumulative_rewards.append(sum_rewards)
        
        print(">>> Evaluation <<<")
        print(f"\tMean Cumulative Rewards: {mean_cumulative_rewards}")
        
        if not self.log:
            return cumulative_rewards, sum_actions