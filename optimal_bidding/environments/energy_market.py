import numpy as np

"""Energy Market Environment"""


class EnergyMarket():
    def __init__(self):
        pass

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

    def sample_state(self, previous_bid, hour):
        """ return the value for the next bid 

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
        nextState = 


class Bid():
    """Bid object so all bids have the same format
    """
    def __init__(self):
        self._agent_id = None

    def power(self):
        pass

