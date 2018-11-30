import matplotlib.pyplot as plt
import pandas as pd
import os
import logging

from agents.expert_agent import ExpertAgent


def run_expert(path_memory_dict):
    expert = ExpertAgent()
    start_date = expert.env.get_start_date()
    step = start_date
    expert.reset_memory_dict()
    done = False
    logging.info('---- Run Expert Policy ----')

    while not done:
        logging.info('---- Step %s ----' % step)
        planned_actions = expert.planning(step)
        done = expert.running(planned_actions)
        step += expert.env._delta_time

    df = pd.DataFrame.from_dict(expert.memory_dict)
    df.to_csv(path_memory_dict)
    logging.info('---- End ----')


def plot_results(path_memory_dict):
    memory_dict = pd.read_csv(path_memory_dict)
    memory_dict = memory_dict.set_index('time_step')

    plt.figure(figsize=(10, 8))
    #plt.plot(memory_dict.index, memory_dict['power_cleared'], label='power')
    plt.plot(memory_dict.index[:48], memory_dict['price_bid'][:48], label='price')
    plt.plot(memory_dict.index[:48], memory_dict['power_cleared'][:48], label='price')
    plt.legend()
    plt.savefig('data/fig.png')


if __name__ == '__main__':
    path_memory_dict = 'data/result_optimization_alone.csv'
    if not os.path.exists(path_memory_dict):
        run_expert(path_memory_dict)
    plot_results(path_memory_dict)