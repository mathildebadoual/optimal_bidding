"""Energy Market Environment"""

import numpy as np
import cvxpy as cvx

from optimal_bidding.environments.grid_scale_battery import Battery


class FCASMarket():
    def __init__(self):
        self._agents_dict = self._create_agents()
        self._num_agents = 6
        self._hour = 0

    def _create_agents(self):
        """Initialize the market
        """
        agents_dict = {}
        # our battery is agent 0
        agents_dict['agent_0'] = Battery
        for i in range(self.num_agents):
            agents_dict['agent_' + str(i)] = Agent()

    def _get_energy(self, hour):
        """From transition matrix
        """
        pass

    def step(self, battery_bid):
        """Collects everyone bids and compute the dispatch
        """
        self.compute_dispatch(battery_bid.type())

    def compute_dispatch(self, bid_type):
        """Here is the optimization problem solving the
        optimal dispatch problem
        """
        power_dispatched = cvx.Variable(self._num_agents)

        power_max = cvx.Parameter(self._num_agents)
        power_min = cvx.Parameter(self._num_agents)
        cost = cvx.Parameter(self._num_agents)
        demand = cvx.Parameter()

        # call bids from agents
        for i, agent_name in enumerate(self._agents_dict.keys()):
            bid = self._agents_dict[agent_name].bid(bid_type=bid_type)
            power_max[i] = bid.power()
            cost[i] = bid.price()

        # build constraints
        constraint = [np.ones(self._num_agents).T * power_dispatched == demand]
        for i in range(self._num_agents):
            constraint += [power_dispatched[i] <= power_max[i]]
            constraint += [power_min[i] <= power_dispatched[i]]

        # build the objective
        objective = cvx.Minimize(power_dispatched.T * cost)

        # build objective
        problem = cvx.Problem(objective, constraint)

        return problem


class Agent():
    """Agent parent class, all the other ch
    """
    def __init__(self):
        pass

    def bid(self, bid_type, time_step=0):
        """Computes the bid at certain time step

        Args:
          bid_type: string 'raise' or 'low'
          time_step: timestamp UTC

        Return:
          bid: Bid object
        """
        raise NotImplementedError


class Agent1(Agent):
    def __init__(self):
        super().__init__()
        self._current_state = None  # TODO(Mathilde): initialize a state here

    def bid(bid_type, time_step=0):
        """Creates a bid using the transition matrix.
        """
        pass


class Bid():
    """Bid object so all bids have the same format
    """
    def __init__(self, power_bid, price_bid):
        self._power_bid = power_bid
        self._price_bid = price_bid
        if self._power_bid <= 0:
            self._bid_type = 'low'
        else:
            self._bid_type = 'raise'

    def power(self):
        return self._power_bid

    def price(self):
        return self._price_bid

    def type(self):
        return self._bid_type
