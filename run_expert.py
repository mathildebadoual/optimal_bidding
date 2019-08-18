import matplotlib.pyplot as plt
import pandas as pd
import os
import logging
import gym
import energym


def run_expert(path_memory_dict):
    env = gym.make('energy_market_battery-v1')
    logging.info('---- Run Expert Policy ----')

    ob = env.reset()
    memory_dict = {'soe': [ob[2]],
                   'power_cleared': [ob[0]],
                   'price_bid': [0],
                   'reward': [0],
                   'done': [0],
                   'time_step': [0],
                   'power_bid': [0],
                   'price_cleared': [0],
                   'power_bid_dqn': [0],
                   'power_bid_expert': [0],
                   'price_bid_dqn': [0],
                   'price_bid_expert': [0],
                   }

    # run until time to re-plan, collect same outputs as the RL agent
    done = False
    while not done:
        ob, reward, done, info_dict = env.step(None)
        memory_dict['soe'].append(ob[2])
        memory_dict['power_cleared'].append(ob[0])
        memory_dict['price_bid'].append(info_dict['action_tot'][1])
        memory_dict['price_cleared'].append(info_dict['price_cleared'])
        memory_dict['reward'].append(reward)
        memory_dict['time_step'].append(info_dict['date'])
        memory_dict['done'].append(done)
        memory_dict['power_bid'].append(info_dict['action_tot'][0])
        memory_dict['power_bid_dqn'].append(info_dict['action_dqn'][0])
        memory_dict['power_bid_expert'].append(info_dict['action_expert'][0])
        memory_dict['price_bid_dqn'].append(info_dict['action_dqn'][1])
        memory_dict['price_bid_expert'].append(info_dict['action_expert'][1])

    df = pd.DataFrame.from_dict(memory_dict)
    df.to_csv(path_memory_dict)
    logging.info('---- End ----')


if __name__ == '__main__':
    path_memory_dict = 'results/expert_alone.csv'
    if not os.path.exists(path_memory_dict):
        run_expert(path_memory_dict)
