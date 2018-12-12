import uuid
import time
import pickle
import sys
import gym.spaces
import tensorflow as tf
from collections import namedtuple
from agents.dqn_utils import *

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])


class QLearner(object):
    def __init__(
            self,
            env,
            q_func,
            optimizer_spec,
            session,
            exploration=LinearSchedule(1000000, 0.1),
            stopping_criterion=None,
            replay_buffer_size=1000000,
            batch_size=32,
            gamma=0.99,
            learning_starts=500,
            learning_freq=4,
            frame_history_len=4,
            target_update_freq=10000,
            grad_norm_clipping=10,
            rew_file=None,
            double_q=True,
            save_path='',
            lander=False,
            with_expert=False):
        """Run Deep Q-learning algorithm.

        You can specify your own convnet using q_func.

        All schedules are w.r.t. total number of steps taken in the environment.

        Parameters
        ----------
        env: gym.Env
            gym environment to train on.
        q_func: function
            Model to use for computing the q function. It should accept the
            following named arguments:
                img_in: tf.Tensor
                    tensorflow tensor representing the input image
                num_actions: int
                    number of actions
                scope: str
                    scope in which all the model related variables
                    should be created
                reuse: bool
                    whether previously created variables should be reused.
        optimizer_spec: OptimizerSpec
            Specifying the constructor and kwargs, as well as learning rate schedule
            for the optimizer
        session: tf.Session
            tensorflow session to use.
        exploration: rl_algs.deepq.utils.schedules.Schedule
            schedule for probability of chosing random action.
        stopping_criterion: (env, t) -> bool
            should return true when it's ok for the RL algorithm to stop.
            takes in env and the number of steps executed so far.
        replay_buffer_size: int
            How many memories to store in the replay buffer.
        batch_size: int
            How many transitions to sample each time experience is replayed.
        gamma: float
            Discount Factor
        learning_starts: int
            After how many environment steps to start replaying experiences
        learning_freq: int
            How many steps of environment to take between every experience replay
        frame_history_len: int
            How many past frames to include as input to the model.
        target_update_freq: int
            How many experience replay rounds (not steps!) to perform between
            each update to the target Q network
        grad_norm_clipping: float or None
            If not None gradients' norms are clipped to this value.
        double_q: bool
            If True, then use double Q-learning to compute target values. Otherwise, use vanilla DQN.
            https://papers.nips.cc/paper/3964-double-q-learning.pdf
        """

        self.target_update_freq = target_update_freq
        self.optimizer_spec = optimizer_spec
        self.batch_size = batch_size
        self.learning_freq = learning_freq
        self.learning_starts = learning_starts
        self.stopping_criterion = stopping_criterion
        self.env = env
        self.test_env = env
        self.session = session
        self.exploration = exploration
        self.rew_file = str(uuid.uuid4()) + '.pkl' if rew_file is None else rew_file
        self.episode_rewards = []
        self.save_path = save_path
        self.q_func = q_func
        self.with_expert = with_expert

        self.mean_rewards_list = []
        self.best_mean_rewards_list = []

        self.loss_list = []

        ###############
        # BUILD MODEL #
        ###############

        input_shape = list(self.env.observation_space.shape)
        input_shape[-1] *= frame_history_len
        self.num_actions = self.env.action_space.n

        # set up placeholders
        # placeholder for current observation (or state)
        self.obs_t_ph = tf.placeholder(
            tf.float32, [None] + input_shape)
        # placeholder for current action
        self.act_t_ph = tf.placeholder(tf.int32, [None])
        # placeholder for current reward
        self.rew_t_ph = tf.placeholder(tf.float32, [None])
        # placeholder for next observation (or state)
        self.obs_tp1_ph = tf.placeholder(
            tf.float32, [None] + input_shape)
        # placeholder for end of episode mask
        # this value is 1 if the next state corresponds to the end of an episode,
        # in which case there is no Q-value at the next state; at the end of an
        # episode, only the current state reward contributes to the target, not the
        # next state Q-value (i.e. target is just rew_t_ph, not rew_t_ph + gamma * q_tp1)
        self.done_mask_ph = tf.placeholder(tf.float32, [None])

        # casting to float on GPU ensures lower data transfer times.
        obs_t_float = tf.cast(self.obs_t_ph, tf.float32) / 255.0
        obs_tp1_float = tf.cast(self.obs_tp1_ph, tf.float32) / 255.0

        # Here, you should fill in your own code to compute the Bellman error. This requires
        # evaluating the current and next Q-values and constructing the corresponding error.
        # TensorFlow will differentiate this error for you, you just need to pass it to the
        # optimizer. See assignment text for details.
        # Your code should produce one scalar-valued tensor: total_error
        # This will be passed to the optimizer in the provided code below.
        # Your code should also produce two collections of variables:
        # q_func_vars
        # target_q_func_vars
        # These should hold all of the variables of the Q-function network and target network,
        # respectively. A convenient way to get these is to make use of TF's "scope" feature.
        # For example, you can create your Q-function network with the scope "q_func" like this:
        # <something> = q_func(obs_t_float, num_actions, scope="q_func", reuse=False)
        # And then you can obtain the variables like this:
        # q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')
        # Older versions of TensorFlow may require using "VARIABLES" instead of "GLOBAL_VARIABLES"
        # Tip: use huber_loss (from dqn_utils) instead of squared error when defining self.total_error
        ######]

        # YOUR CODE HERE
        # Actual Q values
        self.q_values = q_func(obs_t_float, self.num_actions, scope="q_func", reuse=False)
        mask = tf.one_hot(self.act_t_ph, depth=self.num_actions, dtype=tf.bool, on_value=True, off_value=False)
        taken_q_values = tf.boolean_mask(self.q_values, mask)

        # Target Q values
        target_q_values = q_func(obs_tp1_float, self.num_actions, scope="target_q_func", reuse=False)
        if double_q:
            q_values_next = q_func(obs_tp1_float, self.num_actions, scope="q_func", reuse=True)
            argmax_q_values = tf.argmax(q_values_next, axis=1)
            mask = tf.one_hot(argmax_q_values, depth=self.num_actions, dtype=tf.bool, on_value=True, off_value=False)
            max_q_values = tf.boolean_mask(target_q_values, mask)
        else:
            max_q_values = tf.reduce_max(target_q_values, axis=1)

        # Loss computing
        self.total_error = tf.reduce_mean(huber_loss(
            taken_q_values -
            tf.stop_gradient((self.rew_t_ph + gamma * max_q_values * (1 - self.done_mask_ph)))
        ))

        # Get parameters from graph variables
        q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')
        target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q_func')

        ######

        # construct optimization op (with gradient clipping)
        self.learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")
        optimizer = self.optimizer_spec.constructor(learning_rate=self.learning_rate, **self.optimizer_spec.kwargs)
        self.train_fn = minimize_and_clip(optimizer, self.total_error,
                                          var_list=q_func_vars, clip_val=grad_norm_clipping)

        # update_target_fn will be called periodically to copy Q network to target Q network
        update_target_fn = []
        for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                   sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_fn.append(var_target.assign(var))
        self.update_target_fn = tf.group(*update_target_fn)

        # construct the replay buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len, lander=lander)
        self.replay_buffer_idx = None

        ###############
        # RUN ENV     #
        ###############
        self.model_initialized = False
        self.num_param_updates = 0
        self.mean_episode_reward = -float('nan')
        self.best_mean_episode_reward = -float('inf')
        self.last_obs = self.env.reset()
        self.log_every_n_steps = 1000

        self.start_time = None
        self.t = 0

        self.saver = tf.train.Saver()

    def stopping_criterion_met(self):
        return self.stopping_criterion is not None and self.stopping_criterion(self.env, self.t)

    def step_env(self):
        ### 2. Step the env and store the transition
        # At this point, "self.last_obs" contains the latest observation that was
        # recorded from the simulator. Here, your code needs to store this
        # observation and its outcome (reward, next observation, etc.) into
        # the replay buffer while stepping the simulator forward one step.
        # At the end of this block of code, the simulator should have been
        # advanced one step, and the replay buffer should contain one more
        # transition.
        # Specifically, self.last_obs must point to the new latest observation.
        # Useful functions you'll need to call:
        # obs, reward, done, info = env.step(action)
        # this steps the environment forward one step
        # obs = env.reset()
        # this resets the environment if you reached an episode boundary.
        # Don't forget to call env.reset() to get a new observation if done
        # is true!!
        # Note that you cannot use "self.last_obs" directly as input
        # into your network, since it needs to be processed to include context
        # from previous frames. You should check out the replay buffer
        # implementation in dqn_utils.py to see what functionality the replay
        # buffer exposes. The replay buffer has a function called
        # encode_recent_observation that will take the latest observation
        # that you pushed into the buffer and compute the corresponding
        # input that should be given to a Q network by appending some
        # previous frames.
        # Don't forget to include epsilon greedy exploration!
        # And remember that the first time you enter this loop, the model
        # may not yet have been initialized (but of course, the first step
        # might as well be random, since you haven't trained your net...)

        #####

        # YOUR CODE HERE
        # Get frame context
        frame_idx = self.replay_buffer.store_frame(self.last_obs)
        recent_observations = self.replay_buffer.encode_recent_observation()


        # Chose the next action to make
        action = self.shielded_epsilon_greedy_policy(recent_observations)
        #action = self.epsilon_greedy_policy(recent_observations)


        # Perform the action
        obs, reward, done, info = self.env.step(action)
        if done:
            obs = self.env.reset()

        self.replay_buffer.store_effect(frame_idx, action, reward, done)
        self.episode_rewards.append(reward)

        self.last_obs = obs

    def epsilon_greedy_policy(self, recent_observations):
        """
        Perform epsilon greedy policy
        :return:
        """
        if np.random.random() < self.exploration.value(self.t) or not self.model_initialized:
            action = self.env.action_space.sample()
        else:
            q_values = self.session.run(self.q_values,
                                        feed_dict={self.obs_t_ph: [recent_observations]})
            action = np.argmax(q_values[0])
        return action

    def shielded_epsilon_greedy_policy(self, recent_observations):
        """
        Perform epsilon greedy policy, with post-posed shielding.
        :return:
        """
        if np.random.random() < self.exploration.value(self.t) or not self.model_initialized:
            action_safe_flag = False
            while not action_safe_flag:
                action = self.env.action_space.sample()
                action_safe_flag = self.env.is_safe(action)
        else:
            q_values = self.session.run(self.q_values,
                                        feed_dict={self.obs_t_ph: [recent_observations]})
            sorted_q_values_indices = np.argsort(q_values[0])[::-1]
            for action in sorted_q_values_indices:
                if self.env.is_safe(action):
                    break
        return action

    def update_model(self):
        ### 3. Perform experience replay and train the network.
        # note that this is only done if the replay buffer contains enough samples
        # for us to learn something useful -- until then, the model will not be
        # initialized and random actions should be taken
        if (self.t > self.learning_starts and
                self.t % self.learning_freq == 0 and
                self.replay_buffer.can_sample(self.batch_size)):
            # Here, you should perform training. Training consists of four steps:
            # 3.a: use the replay buffer to sample a batch of transitions (see the
            # replay buffer code for function definition, each batch that you sample
            # should consist of current observations, current actions, rewards,
            # next observations, and done indicator).
            # 3.b: initialize the model if it has not been initialized yet; to do
            # that, call
            #    initialize_interdependent_variables(self.session, tf.global_variables(), {
            #        self.obs_t_ph: obs_t_batch,
            #        self.obs_tp1_ph: obs_tp1_batch,
            #    })
            # where obs_t_batch and obs_tp1_batch are the batches of observations at
            # the current and next time step. The boolean variable model_initialized
            # indicates whether or not the model has been initialized.
            # Remember that you have to update the target network too (see 3.d)!
            # 3.c: train the model. To do this, you'll need to use the self.train_fn and
            # self.total_error ops that were created earlier: self.total_error is what you
            # created to compute the total Bellman error in a batch, and self.train_fn
            # will actually perform a gradient step and update the network parameters
            # to reduce total_error. When calling self.session.run on these you'll need to
            # populate the following placeholders:
            # self.obs_t_ph
            # self.act_t_ph
            # self.rew_t_ph
            # self.obs_tp1_ph
            # self.done_mask_ph
            # (this is needed for computing self.total_error)
            # self.learning_rate -- you can get this from self.optimizer_spec.lr_schedule.value(t)
            # (this is needed by the optimizer to choose the learning rate)
            # 3.d: periodically update the target network by calling
            # self.session.run(self.update_target_fn)
            # variable self.num_param_updates useful for this (it was initialized to 0)
            #####

            # YOUR CODE HERE
            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = self.replay_buffer.sample(self.batch_size)

            if not self.model_initialized:
                initialize_interdependent_variables(self.session, tf.global_variables(), {
                    self.obs_t_ph: obs_batch,
                    self.obs_tp1_ph: next_obs_batch,
                })
                self.save_model()
                self.model_initialized = True

            loss, _ = self.session.run((self.total_error, self.train_fn), feed_dict={
                self.obs_t_ph: obs_batch,
                self.act_t_ph: act_batch,
                self.rew_t_ph: rew_batch/10**4,
                self.obs_tp1_ph: next_obs_batch,
                self.done_mask_ph: done_mask,
                self.learning_rate: self.optimizer_spec.lr_schedule.value(self.t)
            })
            self.num_param_updates += 1
            if self.num_param_updates % self.target_update_freq == 0:
                self.session.run(self.update_target_fn)
                print("loss at time step {}: ".format(self.t), loss)
                self.loss_list.append(loss)

        self.t += 1

    def log_progress(self):
        episode_rewards = self.episode_rewards

        if len(episode_rewards) > 0:
            self.mean_episode_reward = np.mean(episode_rewards[-100:])

        if len(episode_rewards) > 100:
            self.best_mean_episode_reward = max(self.best_mean_episode_reward, self.mean_episode_reward)

        if self.t % self.log_every_n_steps == 0 and self.model_initialized:
            print("Timestep %d" % (self.t,))
            print("mean reward (100 episodes) %f" % self.mean_episode_reward)
            print("best mean reward %f" % self.best_mean_episode_reward)
            print("episodes %d" % len(episode_rewards))
            print("exploration %f" % self.exploration.value(self.t))
            print("learning_rate %f" % self.optimizer_spec.lr_schedule.value(self.t))
            if self.start_time is not None:
                print("running time %f" % ((time.time() - self.start_time) / 60.))

            self.save_model()

            self.start_time = time.time()

            sys.stdout.flush()

            self.mean_rewards_list.append(self.mean_episode_reward)
            self.best_mean_rewards_list.append(self.best_mean_episode_reward)

            with open(self.rew_file, 'wb') as f:
                pickle.dump((self.mean_rewards_list, self.best_mean_rewards_list, self.t, self.loss_list), f, pickle.HIGHEST_PROTOCOL)

    def save_model(self):
        path = self.saver.save(self.session, self.save_path)
        print("Model saved in path: %s" % path)

    def test_model(self, test=False, start_date=None):
        if test == False:
            env = self.env
        else:
            env = self.test_env
        start_date = start_date or env._start_date

        save_dict = {
            'power_bid': [],
            'price_bid': [],
            'soc': [],
            'power_cleared': [],
            'reward': [],
            'date': [],
            'price_cleared': [],
            'ref_price': [],
            'power_dqn': [],
            'cost_dqn': [],
        }
        done = False
        obs = env.reset(start_date=start_date)
        list_obs = [np.zeros((3,), dtype=np.float32)] * 23 + [obs]
        save_dict['date'].append(env._date)
        save_dict['soc'].append(obs[1])
        save_dict['power_cleared'].append(obs[0])
        save_dict['power_bid'].append(0)
        save_dict['price_bid'].append(0)
        save_dict['reward'].append(0)
        save_dict['price_cleared'].append(0)
        save_dict['ref_price'].append(0)
        save_dict['power_dqn'].append(0)
        save_dict['cost_dqn'].append(0)
        while not done:
            save_dict['date'].append(env._date)
            action = self.get_action_todo(list_obs)
            obs, reward, done, info = env.step(action)

            # append most recent observation, suppress oldest observation
            # (it's important to respect the same order as in the training)
            list_obs.append(obs)
            list_obs[0:1] = []

            if self.with_expert:
                power, cost = info['action_tot']
                power_dqn, cost_dqn = env.discrete_to_continuous_action(action)
                save_dict['power_dqn'].append(power_dqn)
                save_dict['cost_dqn'].append(cost_dqn)
            else: 
                power, cost = env.discrete_to_continuous_action(action)
            save_dict['power_bid'].append(power)
            save_dict['price_bid'].append(cost)
            
            save_dict['soc'].append(obs[1])
            save_dict['power_cleared'].append(obs[0])
            save_dict['reward'].append(reward)
            save_dict['price_cleared'].append(info['price_cleared'])
            save_dict['ref_price'].append(info['ref_price'])
        return save_dict

    def get_action_todo(self, list_obs):
        """
        :param list_obs: list or numpy array of size (ob_size * history_length)
        :return: int
        """
        q_values = self.session.run(self.q_values,
                                    feed_dict={self.obs_t_ph: [list_obs]})
        action = np.argmax(q_values[0])
        return action