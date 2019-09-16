import numpy as np
from optimal_bidding.utils.data_postprocess import get_demand, get_energy_price, get_energy_price_day_ahead, get_energy_demand_day_ahead

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
        self.total_capacity = 129  # MWh
        self.max_power = 100  # MW
        self.efficiency = 0.98  # percent
        self.min_power = - 100  # MW
        self._energy = 50       
        self.ratio_fcast = 0.8

    def step(self, cleared_power, fcast_cleared_power):
        # add power used for the energy market
        self._energy += self.efficiency * cleared_power
        # add power used for the fcast market
        self._energy += self.ratio_fcast * fcast_cleared_power

    def get_energy(self):
        return self._energy

    def bid(self, energy_price):
        """Overwrite function bid of Agent
        Will create an optimal bid depending on the strategy

        Return:
          bid: Bid object
        """

        return Bid(10, 0)


class AgentRandom(Agent):
    def __init__(self):
        super().__init__()
        self._random_power = 500 + np.random.random_sample() * 100
        self._random_price = np.random.random_sample() * 1000

    def bid(self, time_step=0):
        """Creates a bid using the transition matrix.
        """
        return Bid(self._random_power, self._random_price)

class AgentBaseload(AgentRandom):
    def bid(self, time_step=0):
        """Creates a bid that is meant to be flat and low. 
        (Do we need this agent in addition to the AgentRandom, given that it will return the 
        same price once that's initialized?)
        """
        return Bid(self._random_power, 500) 

class AgentNG(AgentRandom):
    """ This agent will take in the expected energy price and invert it, and multiply it by their bid
    Meant to simulate the behavior of an agent who is going to bid more on the frequency market when energy
    price is low (meaning that it won't be called on). Needs to inherit timestep
    """
    def __init__(self, timestamp):
        super().__init__()
        self.timestamp = timestamp

    def bid(self, timestamp = self.timestamp):
        energy_price = get_energy_price(timestamp)
        energy_prices = get_energy_price_day_ahead(timestamp=timestamp)
        max_energy_price = max(energy_prices) 
        inverted_multiplier = 1 - energy_price/max_energy_price
        return Bid(self._random_power, self._random_price*inverted_multiplier)

class AgentNG2(AgentRandom):
    """ This agent will take in the expected energy demand and invert it, and multiply it by its bid
    Meant to simulate the behavior of an agent who is going to bid more on the frequency market when energy
    demand is low (meaning that it won't be called on). Needs to inherit timestep
    """
    def __init__(self, timestamp):
        super().__init__()
        self.timestamp = timestamp

    def bid(self, timestamp = self.timestamp):
        energy_demand = get_energy_demand(timestamp)
        energy_demands = get_energy_demand_day_ahead(timestamp=timestamp)
        max_energy_demand = max(energy_demands) 
        inverted_multiplier = 1 - energy_demand/max_energy_price
        return Bid(self._random_power*inverted_multiplier, self._random_price)


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
