"""Energy Market Environment"""

import numpy as np
import cvxpy as cvx
import pandas as pd

from optimal_bidding.environments.agents import AgentDeterministic, Bid
import optimal_bidding.utils.data_postprocess as data_utils


class FCASMarket():
    def __init__(self):
        self._num_agents = 10
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

    def get_timestamp(self):
        return self._timestamp

    def _create_agents(self):
        """Initialize the market agents

        Args:
          None

        Return:
          agent_dict: dictionary of the agents
        """
        agents_dict = {}

        agents_dict['agent_0'] = AgentDeterministic(5, 5)
        agents_dict['agent_1'] = AgentDeterministic(9, 7)
        agents_dict['agent_2'] = AgentDeterministic(11, 2)
        agents_dict['agent_3'] = AgentDeterministic(15, 7)
        agents_dict['agent_3'] = AgentDeterministic(19, 5)
        agents_dict['agent_4'] = AgentDeterministic(20, 6)
        agents_dict['agent_7'] = AgentDeterministic(83, 30)
        agents_dict['agent_8'] = AgentDeterministic(100, 1000)

        return agents_dict

    def step(self, battery_bid):
        """Collects everyone bids and compute the dispatch
        """
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
        print(power_dispatched)
        power_max = cvx.Parameter(self._num_agents)
        power_min = cvx.Parameter(self._num_agents)
        cost = cvx.Parameter(self._num_agents)
        demand = cvx.Parameter()

        # get data from AEMO file
        if battery_bid.type() == 'load':
            demand = data_utils.get_low_demand(self._timestamp)
        else:
            demand = data_utils.get_raise_demand(self._timestamp)

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
        print(constraint)
        # build objective
        problem = cvx.Problem(objective, constraint)

        # solve problem
        problem.solve(verbose=True)
        print(battery_bid.price())
        print(battery_bid.power_signed())

        print(power_dispatched.value)
        # get the power cleared for the battery
        power_cleared = power_dispatched.value[-1]

        # for i, name_agent in enumerate(self._agents_dict.keys()):
        #     print('agent_name: %s' % name_agent)
        #     print('power_dispatch: %s' % power_dispatched.value[i])
        #     print('cost: %s' % cost_np[i])

        # compute clearing price
        possible_costs = []
        for i, power in enumerate(power_dispatched.value):
            if abs(power) > 1e-2:
                possible_costs.append(cost.value[i])
        clearing_price = np.max(possible_costs)

        # print('clearing_price: %s' % clearing_price)

        battery_bid_cleared = Bid(power_cleared, battery_bid.price(), bid_type=battery_bid.type())

        return battery_bid_cleared, clearing_price
