"""Energy Market Environment"""

import numpy as np
import cvxpy as cvx
import pandas as pd

from optimal_bidding.environments.agents import Battery, AgentRandom
from optimal_bidding.utils.data_postprocess import get_demand, get_energy_price


class FCASMarket():
    def __init__(self):
        self._num_agents = 6
        self._agents_dict = self._create_agents()
        self._start_timestamp = pd.Timestamp(year=2018,
                                             month=6,
                                             day=1,
                                             hour=4,
                                             minute=30)
        self._end_timestamp = pd.Timestamp(year=2018,
                                           month=7,
                                           day=1,
                                           hour=4,
                                           minute=30)
        self._timestamp = self._start_timestamp

    def _create_agents(self):
        """Initialize the market agents

        Args:
          None

        Return:
          agent_dict: dictionary of the agents
        """
        agents_dict = {}
        # our battery is agent 0
        self._battery = Battery()
        for i in range(self._num_agents - 1):
            agents_dict['agent_' + str(i)] = AgentRandom()
        return agents_dict

    def step(self):
        """Collects everyone bids and compute the dispatch
        """
        battery_bid = self._battery.bid(get_energy_price(self._timestamp))
        power, clearing_price = self.compute_dispatch(battery_bid)
        self._battery.step(power, clearing_price)
        self._timestamp += pd.Timedelta('30 min')
        if self._timestamp > self._end_timestamp:
            return False
        return True

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

        # get data from AEMO file
        demand = get_demand(self._timestamp)

        # call bids from agents
        power_max_np = np.zeros(self._num_agents)
        cost_np = np.zeros(self._num_agents)
        for i, agent_name in enumerate(self._agents_dict.keys()):
            bid = self._agents_dict[agent_name].bid()
            power_max_np[i] = bid.power()
            cost_np[i] = bid.price()

        # add battery bid
        power_max_np[-1] = battery_bid.power()
        cost_np[-1] = battery_bid.price()

        power_max.value = power_max_np
        cost.value = cost_np
        power_min.value = np.zeros(self._num_agents)

        # build constraints
        constraint = [np.ones(self._num_agents).T * power_dispatched == demand]
        for i in range(self._num_agents):
            constraint += [power_dispatched[i] <= power_max[i]]
            constraint += [power_min[i] <= power_dispatched[i]]

        # build the objective
        objective = cvx.Minimize(power_dispatched.T * cost)

        # build objective
        problem = cvx.Problem(objective, constraint)

        # solve problem
        problem.solve(verbose=False)

        # get the power cleared for the battery
        power_cleared = power_dispatched.value[-1]

        # compute clearing price
        possible_costs = []
        for i, power in enumerate(power_dispatched.value):
            if power > 1e-5:
                possible_costs.append(cost.value[i])
        clearing_price = np.max(possible_costs)

        return power_cleared, clearing_price
