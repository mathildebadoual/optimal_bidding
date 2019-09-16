import numpy as np
import optimal_bidding.utils.data_postprocess as data
import cvxpy as cvx



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
        self._total_capacity = 129  # MWh
        self._max_power = 70  # MW
        self._efficiency = 1  # percent
        self._init_energy = 0
        self._ratio_fcast = 0.8
        self._max_ramp_power = 50  # MW

        # for optimization
        self._horizon = 48  # steps so 24 hours

    def reset(self):
        self._init_energy = np.random.random_sample() * self._total_capacity

    def step(self, cleared_power, fcast_cleared_power):
        # add power used for the energy market
        new_energy = self._energy + self._efficiency * cleared_power + \
                self._ratio_fcast * fcast_cleared_power
        if new_energy > self._max_power:
            self._energy = self._max_power
        if new_energy < self._min_power:
            self._energy = self._min_power
        else:
            self._energy = new_energy

    def get_energy(self):
        return self._energy

    def bid(self, timestamp=0):
        """Overwrite function bid of Agent
        Will create an optimal bid depending on the strategy

        Return:
          bid: Bid object
        """

        energy_price = data.get_energy_price_day_ahead(timestamp, horizon=self._horizon)
        low_price = data.get_low_price_day_ahead(timestamp, horizon=self._horizon)
        raise_price = data.get_raise_price_day_ahead(timestamp, horizon=self._horizon)

        p_gen, p_load, s_raise, s_low = self._solve_optimal_bidding_mpc(
                energy_price,
                low_price,
                raise_price,
                )

        if s_raise[0] > 10e-5:
            return Bid(s_raise[0], raise_price[0])
        else:
            return Bid(s_low[0], low_price[0])

    def _solve_optimal_bidding_mps(self, energy_price, low_price, raise_price):
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
        s_low = cvx.Variable(self._horizon)
        p_gen = cvx.Variable(self._horizon)
        p_load = cvx.Variable(self._horizon)
        soe = cvx.Variable(self._horizon)
        m = cvx.Variable(self._horizon, boolean=True)

        # constraints 
        constraints = []

        for t in range(self._horizon - 1):
            constraints += [
            soe[t+1] == soe[t] - self._efficiency * (p_gen[t] + s_raise[t]) + \
             self._efficiency * (p_load[t] + s_low[t])]

        constraints += [0 <= p_load,
             p_load <= m * self._max_power,
             0 <= p_gen,
             p_gen <= (1 - m) * self._max_power,
             0 <= s_low,
             s_low <= m * self._max_power,
             0 <= s_raise,
             s_raise <= (1 - m) * self._max_power,
             0 <= soe,
             soe <= self._total_capacity,
             ]

        # objective
        objective = cvx.Minimize(
            energy_price * p_load - energy_price * (p_gen + self._ratio_fcast * s_raise) + \
            raise_price * s_raise + low_price * s_low)

        # solve problem
        problem = cvx.Problem(objective, constraints)
        problem.solve(verbose=True)

        return p_gen.value, p_load.value, s_raise.value, s_low.value


class AgentRandom(Agent):
    def __init__(self):
        super().__init__()
        self._random_power = 500 + np.random.random_sample() * 100
        self._random_price = np.random.random_sample() * 1000

    def bid(self, timestamp=0):
        """Creates a bid using the transition matrix.
        """
        return Bid(self._random_power, self._random_price)


class AgentBaseload(AgentRandom):
    def bid(self, timestamp=0):
        """Creates a bid that is meant to be flat and low. 
        (Do we need this agent in addition to the AgentRandom, given that it will return the 
        same price once that's initialized?)
        """
        return Bid(self._random_power, 500) 
 

class AgentNaturalGas(AgentRandom):
    """ This agent will take in the expected energy price and invert it, and multiply it by their bid
    Meant to simulate the behavior of an agent who is going to bid more on the frequency market when energy
    price is low (meaning that it won't be called on). Needs to inherit timestep
    """
    def __init__(self):
        super().__init__()
        self._horizon = 48

    def bid(self, timestamp=0):
        energy_price = data.get_energy_price(timestamp)
        energy_prices = data.get_energy_price_day_ahead(timestamp, horizon=self._horizon)
        max_energy_price = max(energy_prices) 
        inverted_multiplier = 1 - energy_price / max_energy_price
        return Bid(self._random_power, self._random_price * inverted_multiplier)


class AgentNaturalGas2(AgentRandom):
    """ This agent will take in the expected energy demand and invert it, and multiply it by its bid
    Meant to simulate the behavior of an agent who is going to bid more on the frequency market when energy
    demand is low (meaning that it won't be called on). Needs to inherit timestep
    """
    def __init__(self, timestamp):
        super().__init__()
        self._horizon = 48

    def bid(self, timestamp=0):
        energy_demand = data.get_energy_demand(timestamp)
        energy_demands = data.get_energy_demand_day_ahead(timestamp, horizon=self._horizon)
        max_energy_demand = max(energy_demands) 
        inverted_multiplier = 1 - energy_demand / max_energy_price
        return Bid(self._random_power * inverted_multiplier, self._random_price)


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
