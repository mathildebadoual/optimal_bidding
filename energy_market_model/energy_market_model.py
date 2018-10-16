import numpy as np
import cvxpy as cvx


# This class gather the two elements of the environment:
# the battery and the market
class Env():
    def __init__(self):
        num_other_agents = 10
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
        self.opt_problem = self.build_problem()
        self.time = time_init
        self.delta_time = delta_time

    def build_problem(self):
        # Parameters and Variables
        self.cost_generators = cvx.Parameter((self.num_agents+1,))
        self.bids_generators = cvx.Parameter((self.num_agents+1,))
        self.demand = cvx.Parameter()
        self.bids_cleared = cvx.Variable((self.num_agents+1,), boolean=True)

        # Constraints and Objective function
        constraint = [np.sum(self.bids_generators) == self.demand]
        objective = cvx.Minimize(
            cvx.kron(self.bids_cleared, self.bids_generators).T *
            self.cost_generators)
        problem = cvx.Problem(objective, constraint)

        return problem

    def reset(self):
        self.time = 0

    def step(self, action):
        # action = [quantity, cost]

        # assign values to the cvxpy parameters
        self.cost_generators.value[1:] = self.cost_generators_base
        self.cost_generators.value[0] = action[1]
        self.bids_generators.value[1:] = self.get_bids_other_generators(
            self.time)
        self.bids_generators.value[0] = action[0]
        self.demand.value = self.get_demand(self.time)

        # solve problem
        self.problem.solve()

        # step in time
        self.time += self.delta_time

        # send result to battery
        self.bids_cleared.value[0]

        # return observation

    def get_bids_other_generators(self, time):
        pass

    def get_delta_time(self, time):
        pass


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
        if self.min_power <= energy_to_add <= self.max_power:
            next_soe = self.soe + energy_to_add
            if self.min_soe <= next_soe <= self.max_soe:
                self.soe = next_soe
        return self.soe
