import tensorflow as tf
import datetime
import gym
import energym
import time
import argparse

import agents.dqn as dqn
from agents.dqn_utils import *


def model(input, num_actions, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        with tf.variable_scope("action_value"):
            out = tf.contrib.layers.fully_connected(out, num_outputs=32,         activation_fn=tf.nn.relu)
            out = tf.contrib.layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out


def create_controller(env,
          session,
          num_timesteps,
          save_path,
          rew_file):

    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0

    lr_multiplier = 1.0
    lr_schedule = PiecewiseSchedule([
                                         (0,                   1e-4 * lr_multiplier),
                                         (num_iterations / 10, 1e-4 * lr_multiplier),
                                         (num_iterations / 2,  5e-5 * lr_multiplier),
                                    ],
                                    outside_value=5e-5 * lr_multiplier)
    optimizer = dqn.OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        kwargs=dict(epsilon=1e-4),
        lr_schedule=lr_schedule
    )

    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return t > 100000

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (5e4, 0.01),
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
        learning_starts=500,
        learning_freq=4,
        frame_history_len=4,
        target_update_freq=1000,
        grad_norm_clipping=10,
        double_q=True,
        save_path=save_path,
    )

    return controller


def test_model(controller):
    controller.test_model()


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
                        help='Path for the rewards file (optional).')
    args = parser.parse_args()

    # ************ MAIN ************
    env = gym.make('energy_market_battery-v0')
    # divide data in test and train


    # initialize
    session = get_session()

    controller = create_controller(env, session, num_timesteps=1e8, save_path='/tmp/model.ckpt', rew_file=args.rew_file)

    # train controller
    while not controller.stopping_criterion_met():
        controller.step_env()
        # at this point, the environment should have been advanced one step (and
        # reset if done was true), and self.last_obs should point to the new latest
        # observation
        controller.update_model()
        controller.log_progress()

    env.close()

    # test controller
    test_model(controller)


if __name__ == "__main__":
    main()
