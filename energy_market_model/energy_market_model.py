import numpy as np
import cvxpy as cvx

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
    def __init__(self, num_other_agents, grid_parameters, index_storage ,time_init=0, delta_time=60):
        self.num_other_agents = num_other_agents
        self.time = time_init
        self.delta_time = delta_time
        self.grid_parameters = grid_parameters
        self.shape_grid = grid_parameters['h'].shape
        self.opt_problem = self.build_opt_problem()
        self.index_storage = index_storage

    def build_opt_problem(self):
        # build parameters
        theta_max = self.grid_parameters['theta_max']
        theta_min = self.grid_parameters['theta_max']
        h = self.grid_parameters['h']
        self.p_max = cvx.Parameter(self.shape_grid[0])
        self.p_min = cvx.Parameter(self.shape_grid[0])
        self.cost = cvx.Parameter(self.num_other_agents + 1)

        # build variables
        self.p = cvx.Variable(self.num_other_agents + 1)
        theta = cvx.Variable(self.shape_grid[0])
        self.cleared = cvx.Variable(self.num_other_agents + 1, boolean=True)

        # build constraints
        agents_in_grid = self.grid_parameters['agents_in_grid']
        grid_connexion = self.grid_parameters['grid_connexion']
        constraint = []
        for i in range(self.shape_grid[0]):
            constraint += [agents_in_grid[i,:] * self.p == grid_connexion[i,:] * ((theta[i] - theta) * h[i,:]).T]
            constraint += [theta[i] <= theta_max[i]] + [theta_min[i] <= theta[i]]

        for k in range(self.num_other_agents + 1):
            constraint += [self.p[k] - self.p_max[k] * self.cleared[k] <= 0] + [self.p_min[k] * self.cleared[k] - self.p[k] <= 0]

        #constraint += [self.cleared[2] == 1]

        # build the objective
        objective = cvx.Minimize(self.p.T * self.cost)

        # build objective
        problem = cvx.Problem(objective, constraint)
        return problem

    def reset(self):
        self.time = 0

    def step(self, action):
        # assign values to the cvxpy parameters
        self.p_min.value, self.p_max.value, self.cost.value = self.get_bids_actors(action, self.time)

        # solve the problem
        self.opt_problem.solve(verbose=True)
        self.time += self.delta_time

        # send result to battery
        return self.p.value, self.cleared.value

    def get_bids_actors(self, action, time):
        p_min = np.array([10, 20, -47.8, 7.6, 11.2, 0, 0, 29.5, -9.0, -3.5, -6.1, -13.8, -14.9, action[0]])
        p_max = np.array([10000, 80, -47.8, 7.6, 11.2, 0, 0, 29.5, -9.0, -3.5, -6.1, -13.8, -14.9, action[0]])
        cost = np.array([2.450, 3.510, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, action[1]])
        return p_min, p_max, cost

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
