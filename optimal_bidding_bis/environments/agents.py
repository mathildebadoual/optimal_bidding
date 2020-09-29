import numpy as np
import pandas as pd
import cvxpy as cvx
from utils.data_postprocess import DataProcessor
from utils.data_postprocess import read_historical_prices
from utils.data_postprocess import get_stats_historical_prices


last_day_of_data = pd.Timestamp(
    year=2018,
    month=11,
    day=1,
    hour=0,
    minute=0,
)
filename = 'FiveMonths2018_30min.csv'
data_utils = DataProcessor(last_day_of_data, filename)


class Agent():
    """Agent parent class, all the other ch
    """

    def __init__(self):
        pass

    def bid(self, timestamp=0):
        """Computes the bid at certain time step

        Args:
          time_step: timestamp UTC

        Return:
          bid: Bid object
        """
        raise NotImplementedError


class Battery(Agent):
    def __init__(self):
        self._total_capacity = 1029  # MWh
        self._max_power = 300  # MW
        self._efficiency = 1  # percent
        self._init_energy = 0

        # for optimization
        self._horizon = 48  # steps so 24 hours

        self._soe = self._init_energy

    def reset(self):
        self._soe = np.random.random_sample() * self._total_capacity

    def step(self, energy_power_cleared):
        # add power used for the energy market
        new_energy = self._soe + self._efficiency * energy_power_cleared
        if new_energy >= self._total_capacity:
            self._soe = self._total_capacity
        elif new_energy < 0:
            self._soe = 0
        else:
            self._soe = new_energy

    def get_soe(self):
        return self._soe

    def bid_mpc(self, timestamp):
        """Overwrite function bid of Agent
        Will create an optimal bid depending on the strategy

        Return:
          bid: Bid object
        """
        energy_price = read_historical_prices(
            timestamp,
            horizon=self._horizon)
        # mean, cov = get_stats_historical_prices(
        #     timestamp,
        #     self._horizon
        # )

        # print(energy_price, low_price, raise_price, raise_demand)
        # m, p_gen, p_load = self._solve_optimal_bidding_mpc_robust(
        #     mean, cov,
        # )
        m, p_gen, p_load = self._solve_optimal_bidding_mpc_simple(
            energy_price,
        )

        alpha = 0

        # price_bid = mean[0]
        price_bid = energy_price[0]

        # create bid for energy market
        if abs(round(m[0])) == 0:
            bid_energy = Bid(p_gen[0], price_bid - alpha,
                             bid_type='gen')
        else:
            bid_energy = Bid(p_load[0], price_bid - alpha,
                             bid_type='load')

        return bid_energy

    def get_energy_power(self):
        return self._energy_power

    def _solve_optimal_bidding_mpc_robust(self, mean, cov):
        """Solve the optimization problem with CVXPY

        Args:
          energy_prices: numpy.Array of size self._horizon

        Return:
          p_gen: numpy.Array of size self._horizon
          p_load: numpy.Array of size self._horizon
          s_raise: numpy.Array of size self._horizon
          s_low: numpy.Array of size self._horizon
        """
        p_gen = cvx.Variable(self._horizon)
        p_load = cvx.Variable(self._horizon)
        p_tot = cvx.Variable(self._horizon)
        soe = cvx.Variable(self._horizon)
        m = cvx.Variable(self._horizon, boolean=True)

        # constraints
        constraints = []

        for t in range(self._horizon - 1):
            constraints += [
                soe[t + 1] == soe[t] + self._efficiency *
                (- p_gen[t] + p_load[t])
            ]

        constraints += [
            soe[0] == self._soe,
            soe <= self._total_capacity,
            0 <= soe,
            0 <= p_load,
            p_load <= m * self._max_power,
            0 <= p_gen,
            p_gen <= (1 - m) * self._max_power,
            p_tot == p_load - p_gen,
        ]

        print(mean)
        # objective
        d = 1
        objective = cvx.Minimize(
            mean @ p_tot + d * cvx.quad_form(p_tot.T, cov))

        # solve problem
        problem = cvx.Problem(objective, constraints)
        problem.solve(solver='MOSEK', verbose=False)

        return m.value, p_gen.value, p_load.value

    def _solve_optimal_bidding_mpc_simple(self, energy_price):
        """Solve the optimization problem with CVXPY

        Args:
          energy_prices: numpy.Array of size self._horizon

        Return:
          p_gen: numpy.Array of size self._horizon
          p_load: numpy.Array of size self._horizon
          s_raise: numpy.Array of size self._horizon
          s_low: numpy.Array of size self._horizon
        """
        p_gen = cvx.Variable(self._horizon)
        p_load = cvx.Variable(self._horizon)
        soe = cvx.Variable(self._horizon)
        m = cvx.Variable(self._horizon, boolean=True)

        # constraints
        constraints = []

        for t in range(self._horizon - 1):
            constraints += [
                soe[t + 1] == soe[t] + self._efficiency *
                (- p_gen[t] + p_load[t])
            ]

        constraints += [
            soe[0] == self._soe,
            soe <= self._total_capacity,
            0 <= soe,
            0 <= p_load,
            p_load <= m * self._max_power,
            0 <= p_gen,
            p_gen <= (1 - m) * self._max_power,
        ]

        # objective
        objective = cvx.Maximize(
            energy_price @ (p_gen - p_load))

        # solve problem
        problem = cvx.Problem(objective, constraints)
        problem.solve(verbose=False)

        return m.value, p_gen.value, p_load.value


class AgentDeterministic(Agent):
    def __init__(self, price, power):
        super().__init__()
        self._power = power
        self._price = price

    def bid(self, timestamp=0):
        """Creates a bid using the transition matrix.
        """
        return Bid(self._power, self._price)


class AgentBaseload(AgentDeterministic):
    def bid(self, timestamp=0):
        """Creates a bid that is meant to be flat and low.
        (Do we need this agent in addition to the AgentDeterministic,
        given that it will return the
        same price once that's initialized?)
        """
        return Bid(self._random_power, 500)


class AgentNaturalGas(AgentDeterministic):
    """ This agent will take in the expected energy price and invert it,
    and multiply it by their bid
    Meant to simulate the behavior of an agent who is
    going to bid more on the frequency market when energy
    price is low (meaning that it won't be called on).
    Needs to inherit timestep
    """

    def __init__(self):
        super().__init__()
        self._horizon = 48

    def bid(self, timestamp=0):
        energy_price = data_utils.get_energy_price(timestamp)
        energy_prices = data_utils.get_energy_price_day_ahead(
            timestamp,
            horizon=self._horizon)
        max_energy_price = max(energy_prices)
        inverted_multiplier = 1 - energy_price / max_energy_price
        return Bid(self._random_power,
                   self._random_price * inverted_multiplier)


class AgentNaturalGas2(AgentDeterministic):
    """ This agent will take in the expected energy
    demand and invert it, and multiply it by its bid
    Meant to simulate the behavior of an agent who
    is going to bid more on the frequency market when energy
    demand is low (meaning that it won't be called on).
    Needs to inherit timestep
    """

    def __init__(self, timestamp):
        super().__init__()
        self._horizon = 48

    def bid(self, timestamp=0):
        energy_demand = data_utils.get_energy_demand(timestamp)
        energy_demands = data_utils.get_energy_demand_day_ahead(
            timestamp, horizon=self._horizon)
        max_energy_demand = max(energy_demands)
        inverted_multiplier = 1 - energy_demand / max_energy_demand
        return Bid(self._random_power * inverted_multiplier,
                   self._random_price)


class Bid():
    """Bid object so all bids have the same format
    """

    def __init__(self, power_bid, price_bid, bid_type='load'):
        self._power_bid = power_bid
        self._price_bid = price_bid
        self._type = bid_type

    def power(self):
        return self._power_bid

    def power_signed(self):
        if self._type == 'gen':
            return - self._power_bid
        else:
            return self._power_bid

    def price(self):
        return self._price_bid

    def type(self):
        return self._type
