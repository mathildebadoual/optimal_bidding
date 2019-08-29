"""Energy Market Environment"""

import numpy as np

from

class EnergyMarket():
    def __init__(self):
        transition_map = TransitionMap()

    def step(self):
        """Collects everyone bids and compute the dispatch
        """
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
        pass

class PVAgent(Agent):
    def __init__(self):
        super().__init__()

    def sample_next_state_from_transition_matrix(self, previous_bid, hour):
        """ return the value for the next bid by sampling from transition matrix

        previous_bid: what the last bid was
        hour: which hour we are sampling for
        """


    #1. load the heat map

        PVTransitionMap = TransitionMap("PV")
        PV_hour_map = PVTransitionMap.get_transition_map_hour[hour]

    #2. Determine the place where it was for the last timestep
        bids = list(PV_hour_map.columns)
        bid_probabilities = PV_hour_map.loc[previous_bid] # need to test this

    #3. Sample a jump to the next state
        nextState = np.random.choice(elements, p=bid_probabilities)
        return nextState

    def state_to_bid(hour):
        ## will fill this out when the Bid class is more filled out
        Bid.power()= sample_generation(hour)

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
    def __init__(self):
        self._agent_id = None

    def power(self):
        pass

