import gym
import energym
import argparse
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import seaborn as sns

from agents.dqn_baseline import learn


def model(inpt, num_actions, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        with tf.variable_scope("action_value"):
            out = tf.contrib.layers.fully_connected(out, num_outputs=32,         activation_fn=tf.nn.relu)
            out = tf.contrib.layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out


def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


def train(env_version, model_path=""):

    if env_version == 1:
        env = gym.make("energy_market_battery-v1")
    else:
        env = gym.make("energy_market_battery-v0")

    if os.path.isfile(model_path):
        load_model = model_path
    else:
        load_model = None

    act = learn(
        env,
        network='mlp',
        seed=None,
        lr=5e-4,
        total_timesteps=1000000,
        buffer_size=50000,
        exploration_fraction=0.9,
        exploration_final_eps=0.1,
        train_freq=1,
        batch_size=32,
        print_freq=1,
        checkpoint_freq=1000,
        checkpoint_path=model_path,
        learning_starts=1000,
        gamma=1.0,
        target_network_update_freq=500,
        prioritized_replay=False,
        prioritized_replay_alpha=0.6,
        prioritized_replay_beta0=0.4,
        prioritized_replay_eps=1e-6,
        param_noise=False,
        callback=callback,
        load_path=load_model,
    )
    print("Saving model to " + model_path)
    act.save(model_path)


def test(env_version, model_path=''):
    if env_version == 1:
        env = gym.make("energy_market_battery-v1")
    else:
        env = gym.make("energy_market_battery-v0")
    env.set_test()
    act = learn(env, network='mlp', total_timesteps=0, load_path=model_path)

    xdata = []
    ydata1 = []
    ydata2 = []
    ydata3 = []
    ydata4 = []
    ydata5 = []
    size_x_plot = 300

    sns.set(style="darkgrid")

    plt.show()

    axes = plt.gca()
    axes.set_xlim(0, size_x_plot)
    axes.set_ylim(-2100, 3100)
    line1, = axes.plot(xdata, ydata1, lw=2, label='soe')
    line2, = axes.plot(xdata, ydata2, lw=2, label='cleared price')
    line3, = axes.plot(xdata, ydata3, lw=2, label='power bid')
    line4, = axes.plot(xdata, ydata4, lw=2, label='reward')
    line5, = axes.plot(xdata, ydata5, lw=2, label='price bid')

    plt.legend(loc='lower left')

    obs, done = env.reset(), False
    episode_rew = 0
    i = 1
    # obs, rew, done, info = env.step(act(obs[None])[0])
    obs, rew, done, info = env.step(action_dqn=None)
    while not done:
        xdata.append(i)
        ydata1.append(obs[2])
        ydata2.append(info['price_cleared']*20)
        ydata3.append(obs[0])
        ydata4.append(rew*0.1)
        ydata5.append(info['price_bid']*20)
        line1.set_data([xdata, ydata1])
        line2.set_data([xdata, ydata2])
        line3.set_data([xdata, ydata3])
        line4.set_data([xdata, ydata4])
        line5.set_data([xdata, ydata5])
        if i >= size_x_plot - 10:
            axes.set_xlim(i - size_x_plot + 15, i + 15)
        episode_rew += rew
        plt.draw()
        plt.pause(1e-17)
        i += 1
        # obs, rew, done, info = env.step(act(obs[None])[0])
        obs, rew, done, info = env.step(action_dqn=None)
    plt.show()
    print("Episode reward", episode_rew)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', '-rf', type=str, default=None,
                        help='path for the model')
    parser.add_argument('--env', '-v', type=int, default=0,
                        help='environment')
    parser.add_argument("--test", "-t", action="store_true")
    args = parser.parse_args()

    model_path = 'saved_model/' + args.model_path + '.pkl'

    if args.test:
        test(args.env, model_path=model_path)
    else:
        train(args.env, model_path=model_path)
