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
import gc


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
        probs = tfp.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample(1)
            #prob = probs.prob(action)
        return action.numpy(), probs.log_prob(action).numpy(), probs.entropy()
        
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
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

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
        k_initializer_1 = tf.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=2)
        k_initializer_2 = tf.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=1)
        #pol_initializer = tf.keras.initializers.Orthogonal(gain=0.01, seed=33)
        pol_initializer = tf.keras.initializers.Orthogonal(gain=0.01, seed=3)
        activ="tanh"
        '''self.actor = keras.Sequential([
            keras.layers.Dense(128, name='actor_dense_1', activation="relu", input_shape=input_shape,
                               #kernel_initializer='random_uniform',
                                bias_initializer=bias_init),
            keras.layers.Dense(128, name='actor_dense_2', activation="relu", 
                               #kernel_initializer='random_uniform', 
                               bias_initializer=bias_init),
            keras.layers.Dense(units=n_outputs, name='actor_dense_output', activation="relu", 
                               #kernel_initializer=pol_initializer, 
                               #bias_initializer=bias_init
                               ),
            keras.layers.Softmax(name='actor_dense_softmax')
        ])
        
        self.actor = keras.Sequential([
            keras.layers.Dense(128, name='actor_dense_1', activation="relu", input_shape=input_shape),
            keras.layers.Dense(128, name='actor_dense_2', activation="relu"),
            keras.layers.Dense(units=n_outputs, name='actor_dense_output', activation="relu")
            ])
        '''
        self.actor = keras.Sequential([
            keras.layers.Dense(128, name='actor_dense_1', activation=activ, input_shape=input_shape,
                               kernel_initializer=k_initializer_1,
                                #bias_initializer=bias_init
                                ),
            keras.layers.Dense(128, name='actor_dense_2', activation=activ, 
                               kernel_initializer=k_initializer_2, 
                               #bias_initializer=bias_init
                               ),
            keras.layers.Dense(n_outputs, name='actor_dense_output', activation="linear",
                               kernel_initializer=pol_initializer, 
                               #bias_initializer=bias_init
                               )
                               ])
        #'''
 
    # critic
    def _build_vfunction(self, input_shape):
        #bias_init = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=19)
        k_initializer_1 = tf.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=100)
        k_initializer_2 = tf.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=101)
        val_initializer = tf.keras.initializers.Orthogonal(gain=1, seed=33)
        activ='tanh'
        '''
        self.critic = keras.Sequential([
            keras.layers.Dense(128, name='critic_dense_1', activation="relu", input_shape=input_shape),
            keras.layers.Dense(128, name='critic_dense_2', activation="relu"),
            keras.layers.Dense(1, name='critic_dense_output')
        ])
        '''
        
        self.critic = keras.Sequential([
            keras.layers.Dense(128, name='critic_dense_1', activation=activ, input_shape=input_shape,
                               kernel_initializer=k_initializer_1#,
                               #bias_initializer=bias_init
                               ),
            keras.layers.Dense(128, name='critic_dense_2', activation=activ,
                               kernel_initializer=k_initializer_2#,
                               #bias_initializer=bias_init
                               ),
            keras.layers.Dense(1, name='critic_dense_output', activation="linear",
                               kernel_initializer=val_initializer
                               )
        ])
        #'''

    @tf.function
    def train_step_ppo(self, obs, actions, adv, probs, returns):

        #actions = tf.constant(actions)
        #with tf.device('/physical_device/CPU:0'):
        with tf.GradientTape(persistent=True) as tape:
            '''                
            tape.watch(returns)
            tape.watch(probs)
            tape.watch(adv)
            '''

            logits = self.actor(obs)
            dist = tfp.distributions.Categorical(logits=logits)
            
            new_probs = dist.log_prob(actions)
            entropy = dist.entropy()
        
            #_, new_probs, entropy = self.get_action(obs, actions)
            new_value = tf.squeeze(self.get_value(obs))

            logdif = new_probs - probs
            ratio = tf.math.exp(logdif)
            clip = tf.clip_by_value(ratio, 1-self.epsilon, 1+self.epsilon)# * adv
            
            loss_a_clip = tf.multiply(clip, adv)
            loss_a = tf.multiply(ratio, adv)

            loss_actor = tf.math.negative(tf.minimum(loss_a_clip, loss_a))
            loss_actor = tf.reduce_mean(loss_actor)
            entropy = tf.reduce_mean(entropy)
            loss_actor = loss_actor - self.c2*entropy

            #loss_value = tf.keras.losses.mse(returns, new_value)
            loss_value = tf.reduce_mean((new_value - returns) ** 2)

            #loss = loss_actor + loss_value

        grads_critic = tape.gradient(loss_value, self.critic.trainable_variables)
        grads_actor = tape.gradient(loss_actor, self.actor.trainable_variables)
#        grads_critic = tape.gradient(loss, self.critic.trainable_variables)
#        grads_actor = tape.gradient(loss, self.actor.trainable_variables)
        self.optimizer_actor.apply_gradients(zip(grads_actor, self.actor.trainable_variables))
        self.optimizer_critic.apply_gradients(zip(grads_critic, self.critic.trainable_variables))

        return loss_actor, loss_value, entropy

    def save_models(self, addstr = ""):
        v = "" if addstr == "" else "-"
        self.actor.save(filepath="models/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/actor" +f"{v}{addstr}" + "/")
        self.critic.save(filepath="models/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/critic" +f"{v}{addstr}" + "/")

    @profile
    def play_one_step(self, envs: gym.vector.VectorEnv, m: memory.Memory, step, observation):
        
        m.obss[step] = observation
        
        '''Returns the selected action and the probability of that action'''
        logits = self.actor(observation, training=False)
        dist = tfp.distributions.Categorical(logits=logits)
        actions = dist.sample(1)
        probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        m.actions[step] = actions
        m.probs[step] = probs
        m.values[step] = self.get_value(observation).numpy().squeeze()
        
        # make the step(s)
        observation, reward, terminated, truncated, info = envs.step(actions.numpy().squeeze().tolist())
        
        #ids_done = [terminated[i] == 1 or truncated[i] == 1 for i in range(len(t_eval_rewards))]
        '''
        tmp = np.logical_or(terminated, truncated)
        if np.count_nonzero(tmp) > 0:
            ids_done = np.argwhere(tmp).flatten()

            for id in ids_done:
                self.m_eval_rewards = utils.incremental_mean(self.m_eval_rewards, self.t_eval_rewards[id], e)
                e += 1
            # reset reward sum for the finished episodes
            self.t_eval_rewards[ids_done] = 0
        '''
        
#                for i in range(np.count_nonzero(ids_done)):
#                    m_eval_rewards = utils.incremental_mean(m_eval_rewards, np.mean(t_eval_rewards[ids_done]), e)
#                    e += 1

        m.rewards[step] = reward
        m.terminateds[step] = terminated
        m.truncateds[step] = truncated

        return observation


    @profile
    def play_n_timesteps(self, envs: gym.vector.VectorEnv, m: memory.Memory, t_timesteps, single_batch_ts, minibatch_size, epochs):

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

        mean_cumulative_rewards = 0
        last_mcr = 0
        r = 0
        

        t_eval_rewards = np.zeros(shape=(envs.num_envs,))
        m_eval_rewards = 0
        e = 0

        # evaluation settings
        eval_frequency = 150
        evaluation = 0 # evaluation counter

        # seeds for evaluation
        eval_n_episodes = 20
        eval_seeds = [int(x) for x in np.random.randint(1, 100000 + 1, size=eval_n_episodes)]
        #eval_env = gym.make('CartPole-v1', render_mode='text', max_timesteps=self.env_max_timesteps)
        eval_env = gym.make('CartPole-v1', render_mode='text')
        

        # prepare the environments for training
        observation, info = envs.reset()
        batch = envs.num_envs * single_batch_ts
        updates = t_timesteps // single_batch_ts
        #m = mem

        for update in tqdm(range(updates)):

            mean_actor_loss = 0
            mean_critic_loss = 0
            mean_entropy = 0
            c = 0

            t_eval_rewards = np.zeros(shape=(envs.num_envs,))
            m_eval_rewards = 0
            e = 0

            # update alpha
            #alpha = 1
            alpha = 1 - (update/updates)

            # update the optimizer's learning rate
            self.optimizer_actor.learning_rate = self.lr_actor * alpha
            self.optimizer_critic.learning_rate = self.lr_critic * alpha

            # update epsilon
            self.epsilon = self.epsilon * alpha

            #next_truncated = np.zeros(shape=(m.truncateds[0]))
            #next_terminated = np.zeros(shape=(m.terminateds[0]))
            
            for t in range(single_batch_ts):
            
                observation = self.play_one_step(envs, m, t, observation)

            #print(f"Mean episode return: {m_eval_rewards:.5f}")
            #print(f"last {last_mcr}")
            #next_val = self.get_value(observation[np.newaxis]).numpy().squeeze()
            next_val = self.get_value(observation).numpy().squeeze()

#            m.rewards = (m.rewards - np.mean(m.rewards)) / np.std(m.rewards)

            # calc advantages
#            m.advantages = np.zeros(shape=(m.timesteps, m.num_envs), dtype=np.float32)
#            m.returns = np.zeros(shape=(m.timesteps, m.num_envs), dtype=np.float32)

#            m.advantages = [utils.calc_adv_list(single_batch_ts-1, 0, m.rewards[:, env_id], m.values[:, env_id], self._gamma, self._lmbda, m.terminateds[:, env_id], m.truncateds[:, env_id]) for env_id in range(m.num_envs)]

            # calculating advantages and putting them into m.advantages
            utils.calc_advantages(single_batch_ts-1, m.advantages, m.rewards, m.values, next_val, self._gamma, self._lmbda, m.terminateds, m.truncateds)
#            m.advantages = np.array([utils.calc_adv_list(single_batch_ts-1, 0, m.rewards[:, env_id], m.values[:, env_id], self._gamma,
#                                                self._lmbda, m.terminateds[:, env_id], m.truncateds[:, env_id]) for env_id in range(m.num_envs)], dtype=np.float32).transpose()

            # calc returns
#            m.returns = np.array([utils.calc_returns(single_batch_ts-1, 0, m.rewards[:, env_id], self._gamma,
#                                            m.terminateds[:, env_id], m.truncateds[:, env_id]) for env_id in range(m.num_envs)], dtype=np.float32).transpose()
            utils.calc_returns(single_batch_ts-1, m.returns, m.rewards, next_val, self._gamma, m.terminateds, m.truncateds)

            m.flatten()

            mb_actor_loss_l = []
            mb_critic_loss_l = []

            bch_ids = np.arange(0, single_batch_ts, 1)

            for epoch in range(epochs):

                np.random.shuffle(bch_ids)

                for start_ind in np.arange(0, single_batch_ts, minibatch_size):

                    end_ind = start_ind + minibatch_size

                    ids = np.arange(start_ind, end_ind, 1)

                    #mb_obs = np.array(m.f_obss[ids])
                    #mb_actions = np.array(m.f_actions[ids])
                    mb_adv = np.array(m.f_advantages[ids])
                    #mb_probs = np.array(m.f_probs[ids])
                    #mb_returns = np.array(m.f_returns[ids])
                    
                    # advantages normalization
                    mb_adv = (mb_adv - np.mean(mb_adv)) / (np.std(mb_adv) + 1e-8)

                    #loss_actor, loss_critic, entropy = self.train_step_ppo(mb_obs, mb_actions, mb_adv, mb_probs, mb_returns)
                    loss_actor, loss_critic, entropy = self.train_step_ppo(m.f_obss[ids],
                                                                           m.f_actions[ids],
                                                                           mb_adv,
                                                                           m.f_probs[ids],
                                                                           m.f_returns[ids])

                    mean_actor_loss = utils.incremental_mean(mean_actor_loss, np.mean(loss_actor), c)
                    mean_critic_loss = utils.incremental_mean(mean_critic_loss, np.mean(loss_critic), c)
                    mean_entropy = utils.incremental_mean(mean_entropy, entropy, c)
                    c += 1
                    

            if update % eval_frequency == 0:
                print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
                evaluation += 1
                #self.evaluate(eval_n_episodes, eval_seeds, evaluation)
                print()
            
            if self.log:
                with self.train_writer.as_default():
                    tf.summary.scalar('loss_actor', data=mean_actor_loss, step=update)
                    tf.summary.scalar('loss_critic', data=mean_critic_loss, step=update)
                    tf.summary.scalar('entropy', data=mean_entropy, step=update)
                    tf.summary.scalar('m_eval_rewards', data=m_eval_rewards, step=update)
                    tf.summary.scalar('lr_critic', data=self.optimizer_critic.learning_rate, step=update)
                    tf.summary.scalar('lr_actor', data=self.optimizer_actor.learning_rate, step=update)
                    tf.summary.scalar('returns mean', data=np.mean(m.f_returns), step=update)
                    tf.summary.scalar('advantages mean', data=np.mean(m.f_advantages), step=update)
                    
                    if update % eval_frequency == 0:
                        print(f"Actor loss: {mean_actor_loss:.5}")
                        print(f"Critic loss: {mean_critic_loss:.5}")
                        print()

            m.reset()

        print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
        print("Training results")
        print()
        print(f"Mean Actor Loss: {mean_actor_loss}")
        print(f"Mean Critic Loss: {mean_critic_loss}")
        print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")

    def evaluate(self, episodes, seeds, eval_code=None):

        # max number of steps for each episode of the simulator.
        # if reached, the environment is TRUNCATED by the TimeLimit wrapper
        #max_steps = 300

        #env = gym.make('CartPole-v1', render_mode='text', max_timesteps=self.env_max_timesteps)
        env = gym.make('CartPole-v1', render_mode='text')

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
            #mean_actions = [utils.incremental_mean(mean_actions[id], actions[id], episode) for id in range(len(actions))]

            mean_cumulative_rewards = utils.incremental_mean(mean_cumulative_rewards, sum_rewards, episode)


        if self.log:
            if eval_code is not None:
                with self.train_writer.as_default():
                    tf.summary.scalar('cumulative_rewards', data=mean_cumulative_rewards, step=eval_code, description="Metric inside training")
                    tf.summary.scalar('chosen_work', data=actions[0], step=eval_code, description="Metric inside training")
                    tf.summary.scalar('chosen_sport', data=actions[1], step=eval_code, description="Metric inside training")
                    tf.summary.scalar('chosen_sociality', data=actions[2], step=eval_code, description="Metric inside training")
            else:
                with self.eval_writer.as_default():
                    tf.summary.scalar('cumulative_rewards', data=mean_cumulative_rewards, step=eval_code, description="Metric inside evaluation")
                    tf.summary.scalar('chosen_work', data=actions[0], step=eval_code, description="Metric inside evaluation")
                    tf.summary.scalar('chosen_sport', data=actions[1], step=eval_code, description="Metric inside evaluation")
                    tf.summary.scalar('chosen_sociality', data=actions[2], step=eval_code, description="Metric inside evaluation")
        else:
            sum_actions.append(actions)
            cumulative_rewards.append(sum_rewards)
        
        print(">>> Evaluation <<<")
        print(f"\tMean Cumulative Rewards: {mean_cumulative_rewards:.3f}")
        
        if not self.log:
            return cumulative_rewards, sum_actions