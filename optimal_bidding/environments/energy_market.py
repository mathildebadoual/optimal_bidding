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

    def compute_bid(self, previous_bid, hour):

    #1. load the heat map 

        PVTransitionMap = TransitionMap()
        PV_hour_map = PVTransitionMap.get_transition_map_hour[hour]

    #2. Determine the place where it was for the last timestep 
        bid_probabilities = PV_hour_map.loc[previous_bid] # need to test this

    #3. Sample a jump to the next state 
        np.random.multinomial(1,bid_probabilities,size=K)


class Bid():
    """Bid object so all bids have the same format
    """
    def __init__(self):
        self._agent_id = None

    def power(self):
        pass

