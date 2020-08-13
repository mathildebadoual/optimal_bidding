"""Energy Market Environment"""

import numpy as np
import cvxpy as cvx
import pandas as pd

from environments.agents import AgentDeterministic, Bid
from utils.data_postprocess import DataProcessor


class EnergyMarket():
    def __init__(self, start_time, end_time):
        self._agents_list = self._create_agents()
        self._num_agents = len(self._agents_list)
        self._start_timestamp = start_time
        self._end_timestamp = end_time
        self._timestamp = self._start_timestamp

        filename = 'FiveMonths2018_30min.csv'
        self.data_utils = DataProcessor(self._end_timestamp, filename)

    def get_timestamp(self):
        return self._timestamp

    def _create_agents(self):
        """Initialize the market agents

        Args:
          None

        Return:
          agent_dict: dictionary of the agents
        """
        agents_dict = []

        agents_dict.append(Bid(200, 5))
        agents_dict.append(Bid(60, 9))
        agents_dict.append(Bid(10, 11))
        agents_dict.append(Bid(70, 15))
        agents_dict.append(Bid(53, 19))
        agents_dict.append(Bid(53, 20))
        agents_dict.append(Bid(60, 22))
        agents_dict.append(Bid(53, 31))
        agents_dict.append(Bid(30, 35))
        agents_dict.append(Bid(13, 41))
        agents_dict.append(Bid(53, 40))
        agents_dict.append(Bid(65, 56))
        agents_dict.append(Bid(25, 65))
        agents_dict.append(Bid(58, 67))
        agents_dict.append(Bid(50, 71))
        agents_dict.append(Bid(40, 75))
        agents_dict.append(Bid(70, 86))
        agents_dict.append(Bid(100, 88))
        agents_dict.append(Bid(40, 90))
        agents_dict.append(Bid(120, 95))
        agents_dict.append(Bid(120, 96))
        agents_dict.append(Bid(20, 98))
        agents_dict.append(Bid(50, 105))
        agents_dict.append(Bid(80, 125))
        agents_dict.append(Bid(100, 158))
        agents_dict.append(Bid(100, 205))
        agents_dict.append(Bid(500, 250))
        return agents_dict

    def step(self, battery_bid):
        """Collects everyone bids and compute the dispatch
        """
        battery_bid = Bid(0, 0)

        battery_bid_cleared, clearing_price = self.compute_dispatch(
            battery_bid)
        self._timestamp += pd.Timedelta('30 min')

        if self._timestamp > self._end_timestamp:
            end = True
        else:
            end = False

        return battery_bid_cleared, clearing_price, end

    def compute_dispatch(self, battery_bid):
        """Here is the optimization problem solving the
        optimal dispatch problem

        Args:
          battery_bid: Bid object, bid from the battery

        Return:
          power_cleared = float
          clearing_price = float
        """
        power_dispatched = cvx.Variable(self._num_agents)
        power_max = cvx.Parameter(self._num_agents)
        power_min = cvx.Parameter(self._num_agents)
        cost = cvx.Parameter(self._num_agents)
        demand = cvx.Parameter()
        # print(battery_bid)

        # get data from AEMO file
        demand = self.data_utils.get_energy_demand(self._timestamp) * 0.6

        # call bids from agents
        power_max_np = np.zeros(self._num_agents)
        cost_np = np.zeros(self._num_agents)
        for i, bid in enumerate(self._agents_list):
            power_max_np[i] = bid.power()
            cost_np[i] = bid.price()

        # add battery bid
        if battery_bid.type() == 'gen':
            power_max_np[-1] = battery_bid.power()
            cost_np[-1] = battery_bid.price()
        else:
            power_max_np[-1] = 0
            cost_np[-1] = 0
            print('battery bid power when load: %s' % battery_bid.power())
            print('demand: %s' % demand)
            demand += battery_bid.power()

        power_max.value = power_max_np
        cost.value = cost_np
        power_min.value = np.zeros(self._num_agents)

        # build constraints
        constraint = [np.ones(self._num_agents).T @ power_dispatched == demand]
        for i in range(self._num_agents):
            constraint += [power_dispatched[i] <= power_max[i]]
            constraint += [power_min[i] <= power_dispatched[i]]

        # build the objective
        objective = cvx.Minimize(power_dispatched.T @ cost)
        # build objective
        problem = cvx.Problem(objective, constraint)

        # solve problem
        problem.solve(verbose=False)

        # get the power cleared for the battery
        if battery_bid.type() == 'gen':
            power_cleared = power_dispatched.value[-1]
        else:
            power_cleared = battery_bid.power()
        # compute clearing price
        possible_costs = [0]
        for i, power in enumerate(power_dispatched.value):
            if abs(power) > 1e-2:
                possible_costs.append(cost.value[i])
        clearing_price = np.max(possible_costs)

        print('clearing_price: %s' % clearing_price)

        battery_bid_cleared = Bid(power_cleared,
                                  battery_bid.price(),
                                  bid_type=battery_bid.type())

        return battery_bid_cleared, clearing_price
