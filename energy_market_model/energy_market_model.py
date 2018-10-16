import numpy as np


# This class gather the two elements of the environment:
# the battery and the market
class Env():
    def __init__(self, num_other_agents):
        self.market_model = MarketModel(num_other_agents)
        self.storage_system = StorageSystem()

    def reset(self):
        self.market_model.reset()
        self.storage_system.reset()

    def step(self, action):
        quantity_cleared, price_cleared = self.market_model.step(action)
        actual_soe = self.storage_system(quantity_cleared)

        # define the state
        state = np.array(quantity_cleared, price_cleared, actual_soe)
        return state


class MarketModel():
    def __init__(self, num_other_agents, time_init=0, delta_time=60):
        self.num_other_agents = num_other_agents
        self.time = time_init
        self.delta_time = delta_time

    def reset(self):
        self.time = 0

    def step(self, action):
        # action = (quantity, cost) is type tuple

        # assign values to the cvxpy parameters
        bids_other = self.get_bids_other_generators(
            self.time)
        bids =  [action] + bids_other
        demand = self.get_demand(self.time)

        # solve problem
        bids.sort(key=lambda tup: tup[1])
        quantity_cleared = 0
        cleared_bids = bids.copy()
        for i, bid in enumerate(bids):
            if quantity_cleared + bid[0] <= demand:
                quantity_cleared += bid[0]
            elif quantity_cleared < demand:
                cleared_bids[i] = (demand - quantity_cleared, bid[1])
            else:
                cleared_bids[i] = (0, bid[1])

        # step in time
        self.time += self.delta_time

        # send result to battery
        return cleared_bids

    def get_bids_other_generators(self, time):
        return [(30, 2), (24, 3), (50, 2.4), (10, 2), (4, 1.9)]

    def get_demand(self, time):
        return 60


class StorageSystem():
    def __init__(self):
        self.max_soe = 100   # MW
        self.min_soe = 0
        self.max_power = 10   # MWh
        self.min_power = 0
        self.efficiency_ratio = 0.99
        self.soe = 0

    def reset(self):
        self.soe = 0

    def initialize(self, soe_init):
        self.soe = soe_init

    def step(self, power):
        energy_to_add = self.efficiency_ratio * power
        if self.min_power <= abs(energy_to_add) <= self.max_power:
            next_soe = self.soe + energy_to_add
            if self.min_soe <= next_soe <= self.max_soe:
                self.soe = next_soe
        return self.soe
