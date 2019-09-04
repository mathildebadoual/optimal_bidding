"""Energy Market Environment"""

import numpy as np
import cvxpy as cvx

from optimal_bidding.utils.data_postprocess import TransitionMap


class EnergyMarket():
    def __init__(self):
        pass

    def step(self):
        """Collects everyone bids and compute the dispatch
        """
        pass

    def compute_dispatch(self):
        pass


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


class PVAgent(Agent):
    def __init__(self):
        super().__init__()
        self.pv_transition_map = TransitionMap("PV")

    def sample_next_state_from_transition_matrix(self, previous_bid, hour):
        """ return the value for the next bid by sampling from transition
        matrix

        Args:
          previous_bid: what the last bid was
          hour: which hour we are sampling for

        Return:
          next_state
        """

        pv_hour_map = self.pv_transition_map.get_transition_map_hour(hour)

        # Determine the place where it was for the last timestep
        bids = list(pv_hour_map.columns)
        bid_probabilities = pv_hour_map.loc[previous_bid]

        # Sample a jump to the next state
        next_state = np.random.choice(elements, p=bid_probabilities)
        return next_state

    def state_to_bid(hour):
        # will fill this out when the Bid class is more filled out
        # Bid.power() = sample_generation(hour)
        pass

    def sample_generation(hour):
        """samples a day of solar generation from a year; see utils
        fuction sample_day_solar_generation(month) for more info

        Currently assumes that we'll stick with a single month.

        """
        generation_curve = sample_day_solar_generation(6)
        return generation_curve["kW"][hour]


class Bid():
    """Bid object so all bids have the same format
    """
    def __init__(self, power_bid, price_bid):
        self._power_bid = power_bid
        self._price_bid = price_bid

    def power(self):
        return self._power_bid

    def price(self):
        return self._price_bid
