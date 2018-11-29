import cvxpy as cvx
import gym
import energym
import numpy as np
import pandas as pd
import os

# The difference between DQN and this expert agent is that it depends on the environment directly


class ExpertAgent(object):
    def __init__(self):
        # this is the environment on which the controller will be applied
        self.env = gym.make('energy_market_battery-v0')

        # we create those environments to get some info (clearing prices + battery dynamic)
        self.market = gym.make('energy_market-v0')
        self.battery = gym.make('battery-v0')

        # to create the prediction prices (perfect forecast)
        self.data_path = 'data'
        self.price_prediction_file_path = os.path.join(self.data_path, "price_prediction.csv")
        self.get_prediction_cleared_prices()
        self.price_prediction_df = self.load_price_predictions()

        # parameters for the online controller
        self.memory_dict = {'soe': [0],
                            'power_cleared': [],
                            'price_cleared': [],
                            'reward': [],
                            'done': [],
                            'time_step': [],
                            }
        self.planning_frequency = 1
        self.time_horizon = 16
        self.max_soe, self.min_soe, self.max_power, self.min_power, self.battery_efficiency = self._battery.get_parameters_battery()

        # create the optimization problem
        self.problem = self.create_optimization_problem()

    def get_prediction_cleared_prices(self):
        # We run the all simulation without the battery (considering we are price take we do not influence the market).
        # This function needs to be called once and then we store the result in a pickle
        if not os.path.exists(self.price_prediction_file_path):
            done = False
            action = np.array([0, 0])
            price_prediction_dict = {'time_step': [], 'values': []}
            while not done:
                ob, reward, done, info_dict = self.market.step(action)
                price_prediction_dict['values'].append(ob[1])
                price_prediction_dict['time_step'].append(info_dict['date'])
            price_prediction_df = pd.DataFrame.from_dict(price_prediction_dict)
            price_prediction_df.to_csv(self.price_prediction_file_path)

    def load_price_predictions(self):
        price_prediction_df = pd.DataFrame.from_csv(self.price_prediction_file_path)
        return price_prediction_df

    def create_optimization_problem(self):
        # create a generic optimization problem solved for planning
        self.price_predictions_interval = cvx.Parameter(self.time_horizon)
        self.initial_soe = cvx.Parameter()

        self.soe = cvx.Variable()
        self.planned_power = cvx.Variable(self.time_horizon)

        opt = cvx.Maximize(self.price_predictions_interval * self.planned_power)

        constraints = [self.soe[0] == self.initial_soe]
        for i in range(self.time_horizon-1):
            constraints += [self.soe[i+1] == self.soe[i] + self.battery_efficiency * self.planned_power[i]]

        constraints += [self.soe <= self.max_soe] + [self.min_soe <= self.soe]
        constraints += [self.planned_power <= self.max_power] + [self.min_power <= self.planned_power]

        return cvx.Problem(opt, constraints)

    def planning(self, step):
        # solve optimization problem from actual time step for a certain horizon
        self.price_predictions_interval.value = self.price_prediction_df[self.price_prediction_df['time_step'] >= step].values[:self.time_horizon]
        self.initial_soe.value = self.memory_dict['soe'][-1]

        self.Problem.solve()
        planned_actions = self.planned_power.value
        return planned_actions

    def running(self, planned_actions):
        # run until time to re-plan, collect same outputs as the RL agent

        for i in range(self.time_horizon):
            if i > self.planning_frequency or done:
                break
            action = planned_actions[i]
            ob, reward, done, info_dict = self.env.step(action)
            self.memory_dict['soe'].append(ob[0])
            self.memory_dict['power_cleared'].append(ob[1])
            self.memory_dict['price_cleared'].append(ob[2])
            self.memory_dict['reward'].append(reward)
            self.memory_dict['time_step'].append(info_dict['date'])


if __name__ == '__main__':
    expert = ExpertAgent()
    start_date = expert.env._start_date
    planned_actions = expert.planning(start_date)
    print(planned_actions)
