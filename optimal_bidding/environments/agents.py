import numpy as np
import pandas as pd
import cvxpy as cvx
from optimal_bidding.utils.data_postprocess import DataProcessor


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
        self._max_ramp = 200
        self._efficiency = 1  # percent
        self._init_energy = 0
        self._ratio_fcast = 0.8
        self._max_ramp_power = 50  # MW

        # for optimization
        self._horizon = 48  # steps so 24 hours

        self._soe = self._init_energy

    def reset(self):
        self._soe = np.random.random_sample() * self._total_capacity

    def step(self, fcast_cleared_power, energy_power_cleared):
        # add power used for the energy market
        new_energy = self._soe + self._efficiency * energy_power_cleared + \
                self._ratio_fcast * fcast_cleared_power
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
        energy_price = data_utils.get_energy_price_day_ahead(timestamp,
                                                       horizon=self._horizon)
        low_price = data_utils.get_low_price_day_ahead(timestamp,
                                                 horizon=self._horizon)
        raise_price = data_utils.get_raise_price_day_ahead(timestamp,
                                                     horizon=self._horizon)
        raise_demand = data_utils.get_raise_demand_day_ahead(timestamp,
                                                     horizon=self._horizon)
        #print(energy_price, low_price, raise_price, raise_demand)
        n, m, p_gen, p_load, s_raise, s_low = self._solve_optimal_bidding_mpc(
            energy_price,
            low_price,
            raise_demand,
        )

        # create bid for energy market
        if abs(round(m[0])) == 0:
            bid_energy = Bid(p_gen[0], energy_price[0], bid_type='gen')
        else:
            bid_energy = Bid(p_load[0], energy_price[0], bid_type='load')

        # create bid for fcas market
        if abs(round(n[0])) == 0:
            bid_fcas = Bid(s_raise[0], raise_demand[0], bid_type='gen')
        else:
            bid_fcas = Bid(s_low[0], low_price[0], bid_type='load')

        return bid_fcas, bid_energy

    def get_energy_power(self):
        return self._energy_power

    def _solve_optimal_bidding_mpc(self, energy_price, low_price, raise_price):
        """Solve the optimization problem with CVXPY

        Args:
          energy_prices: numpy.Array of size self._horizon
          low_price: numpy.Array of size self._horizon
          raise_price: numpy.Array of size self._horion

        Return:
          p_gen: numpy.Array of size self._horizon
          p_load: numpy.Array of size self._horizon
          s_raise: numpy.Array of size self._horizon
          s_low: numpy.Array of size self._horizon
        """
        s_raise = cvx.Variable(self._horizon)
        p_gen = cvx.Variable(self._horizon)
        p_load = cvx.Variable(self._horizon)
        soe = cvx.Variable(self._horizon)
        m = cvx.Variable(self._horizon, boolean=True)

        # constraints
        constraints = []

        for t in range(self._horizon - 1):
            constraints += [
                soe[t + 1] == soe[t] + self._efficiency *
                (- p_gen[t] - self._ratio_fcast * s_raise[t] + p_load[t])
            ]

        constraints += [
            soe[0] == self._soe,
            soe <= self._total_capacity,
            0 <= soe,
            0 <= p_load,
            p_load <= m * self._max_power,
            0 <= p_gen,
            p_gen <= (1 - m) * self._max_power,
            0 <= s_raise,
            s_raise <= self._max_ramp,
        ]

        # objective
        objective = cvx.Maximize(
                energy_price * (p_gen - p_load + self._ratio_fcast * s_raise) + 0.9 * raise_price * s_raise)

        # solve problem
        problem = cvx.Problem(objective, constraints)
        problem.solve()

        return [0.0], m.value, p_gen.value, p_load.value, s_raise.value, None


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
        energy_prices = data_utils.get_energy_price_day_ahead(timestamp,
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
