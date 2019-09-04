import numpy as np
from optimal_biddin.environment import Agent


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

    def bid(self):
        """Overwrite function bid of Agent
        Will create an optimal bid depending on the strategy

        Return:
          bid: Bid object
        """
        pass
