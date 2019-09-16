import numpy as np
import cvxpy as cvx


class Agent():
    """Agent parent class, all the other ch
    """
    def __init__(self):
        pass

    def bid(self, time_step=0):
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
        self._max_power = 100  # MW
        self._efficiency = 0.98  # percent
        self._min_power = - 100  # MW
        self._energy = 50
        self._ratio_fcast = 0.8

        # for optimization
        self._horizon = 48  # steps so 24 hours

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

    def bid(self, energy_price):
        """Overwrite function bid of Agent
        Will create an optimal bid depending on the strategy

        Return:
          bid: Bid object
        """

        self._solve_optimal_bidding_mpc(
                energy_price,
                low_price,
                raise_price
                )

        return Bid(10, 0)

    def _solve_optimal_bidding_mps(self, energy_price, low_price, raise_price):
        s_raise = cvx.Variable(self._horizon)
        s_low = cvx.Variable(self._horizon)
        p_gen = cvx.Variable(self._horizon)
        p_load = cvx.Variable(self._horizon)




class AgentRandom(Agent):
    def __init__(self):
        super().__init__()
        self._random_power = 500 + np.random.random_sample() * 100
        self._random_price = np.random.random_sample() * 1000

    def bid(self, time_step=0):
        """Creates a bid using the transition matrix.
        """
        return Bid(self._random_power, self._random_price)


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
