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


class Bid():
    """Bid object so all bids have the same format
    """
    def __init__(self):
        self._agent_id = None

    def power(self):
        pass

