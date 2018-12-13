import tensorflow as tf
import datetime
import gym
import energym
import time
import argparse
import pickle
import pandas as pd

import agents.dqn as dqn
from agents.dqn_utils import *


HISTORY_LENGTH = 4


def model(inpt, num_actions, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        with tf.variable_scope("action_value"):
            out = tf.contrib.layers.fully_connected(out, num_outputs=32,         activation_fn=tf.nn.relu)
            out = tf.contrib.layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out


def rnn_model(inpt, num_actions, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        r_inpt = tf.reshape(inpt, shape=(-1, HISTORY_LENGTH, 3))  # BATCH_SIZE * HISTORY_LENGTH * ob_size

        # dense layer
        # r_inpt size: BATCH_SIZE x HISTORY_LENGTH x 64
        r_inpt = tf.layers.dense(r_inpt, 16, activation=tf.nn.leaky_relu, use_bias=True)

        # RNN
        gru_cell = tf.nn.rnn_cell.GRUCell(32, activation=tf.tanh, reuse=None)
        history = HISTORY_LENGTH
        cell_outputs = []
        h = gru_cell.zero_state(tf.shape(r_inpt)[0], dtype=tf.float32)
        for i in range(history):
            cell_output, h = gru_cell(r_inpt[:, i, :], h, scope=scope)
            cell_outputs.append(cell_output)

        cell_outputs = tf.transpose(tf.stack(cell_outputs), perm=[1, 0, 2])

        # Output layer
        flatten_cell_outputs = tf.layers.flatten(cell_outputs)
        out = tf.layers.dense(flatten_cell_outputs, units=num_actions, activation=None, use_bias=False)

        return out


def create_controller(env,
          session,
          num_timesteps,
          save_path,
          rew_file):

    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0

    lr_multiplier = 0.1
    lr_schedule = PiecewiseSchedule([
                                         (0,                   1e-4 * lr_multiplier),
                                         (num_iterations / 10, 1e-4 * lr_multiplier),
                                         (num_iterations / 2,  5e-5 * lr_multiplier),
                                    ],
                                    outside_value=5e-5 * lr_multiplier)
    optimizer = dqn.OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        kwargs=dict(epsilon=1e-3),
        lr_schedule=lr_schedule
    )

    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return t > num_timesteps

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (5e4, 0.1),
            (num_iterations / 2, 0.01),
        ], outside_value=0.01
    )

    controller = dqn.QLearner(
        env=env,
        q_func=model,
        optimizer_spec=optimizer,
        session=session,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        rew_file=rew_file,
        replay_buffer_size=1000000,
        batch_size=32,
        gamma=0.95,
        learning_starts=5000,
        learning_freq=4,
        frame_history_len=HISTORY_LENGTH,
        target_update_freq=500,
        grad_norm_clipping=10,
        double_q=True,
        save_path=save_path,
    )

    return controller


def test_model(controller):
    return controller.test_model()


def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']


def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    session = tf.Session(config=tf_config)
    print("AVAILABLE GPUS: ", get_available_gpus())
    return session


def main():
    # ************ ARGPARSE ************
    parser = argparse.ArgumentParser()
    parser.add_argument('--rew_file', '-rf', type=str, default=None,
                        help='Path for the rewards file and name of the model.')
    parser.add_argument("--test", "-t", action="store_true")
    args = parser.parse_args()
    rew_file = args.rew_file
    rew_name = rew_file.split(".")[0]
    # ************ MAIN ************
    start_time = datetime.datetime.now()
    start_time = [start_time.day,start_time.hour,
                  start_time.minute, start_time.second]

    env = gym.make('energy_market_battery-v0')
    # divide data in test and train


    # initialize
    session = get_session()

    controller = create_controller(env, session, num_timesteps=1e8,
                                   save_path='/tmp/{}.ckpt'.format(rew_name),
                                   rew_file=rew_file)

    # train controller
    if not args.test:
        try:
            while not controller.stopping_criterion_met():
                    controller.step_env()
                    controller.update_model()
                    controller.log_progress()
        except KeyboardInterrupt:
            print("KeyboardInterrupt error caught")
    else:
        controller.saver.restore(controller.session, '/tmp/{}.ckpt'.format(rew_name))
        # FixMe: Ugly
        controller.env._energy_market._gen_df = pd.read_pickle("data/gen_caiso.pkl")
        controller.env._energy_market._dem_df = pd.read_pickle("data/dem_caiso.pkl")

    env.close()

    # test controller
    save_dict = test_model(controller)

    # save result
    with open('results/{}_{}_{}_{}_{}.pkl'.format(*start_time, rew_name), 'wb') as pickle_file:
        pickle.dump(save_dict, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()