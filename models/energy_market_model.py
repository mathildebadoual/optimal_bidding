import numpy as np
import cvxpy as cvx
from pyiso import client_factory
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import pytz


class EmptyDataException(Exception):
    def __init__(self):
        super().__init__()

class OptimizationException(Exception):
    def __init__(self):
        super().__init__()


# This class gather the two elements of the environment:
# the battery and the market
class Env():
    action_space = np.array((2500,), dtype=int)
    observation_space = np.array((3,), dtype=int)

    def __init__(self, num_agents, start_date):
        self.market_model = MarketModel(num_agents, start_date)
        self.storage_system = StorageSystem()
        self.start_date = start_date

        # Discrete quantities metadata
        self.min_quantity, self.max_quantity = 0, 200
        self.min_price, self.max_price = 0, 20
        self.n_discrete_price = 50
        self.n_discrete_quantity = 50

    @property
    def quantity_precision(self):
        return (self.max_quantity - self.min_quantity) / self.n_discrete_quantity

    @property
    def price_precision(self):
        return (self.max_price - self.min_price) / self.n_discrete_price

    def reset(self):
        self.market_model.reset(self.start_date)
        self.storage_system.reset()
        return np.array((0, 0, 0))

    def step(self, discrete_action):
        action = self._discrete_to_continuous_action(discrete_action)
        try:
            quantity_cleared, price_cleared = self.market_model.step(action)
        except OptimizationException:
            return 0, 0, False, None
        except EmptyDataException:
            # import ipdb;ipdb.set_trace()
            print("end of data, resetting the environment...")
            return 0, 0, True, None
        actual_soe, penalty = self.storage_system.step(quantity_cleared)

        # define state and reward
        state = np.array((quantity_cleared, price_cleared, actual_soe))
        reward = quantity_cleared * price_cleared - penalty
        done = False
        info = None

        return state, reward, done, info

    def render(self):
        return

    def close(self):
        return

    def seed(self, seed=None):
        return

    @property
    def unwrapped(self):
        return self

    def _discrete_to_continuous_action(self, discrete_action):
        """
        :param discrete_action: int
        :return: (float, float)
        """
        # maps the integer discrete_action to the grid (quantity, price)
        quantity = (discrete_action % self.n_discrete_quantity) * self.quantity_precision
        cost = (discrete_action // self.n_discrete_quantity) * self.price_precision

        return quantity, cost

class MarketModel():
    def __init__(self, num_agents, start_date, delta_time=datetime.timedelta(hours=1)):
        self.num_agents = num_agents
        self.date = start_date
        self.delta_time = delta_time
        self.opt_problem = self.build_opt_problem()
        self.gen_df = pd.read_pickle("gen_caiso.pkl")
        self.dem_df = pd.read_pickle("dem_caiso.pkl")
        self.timezone = pytz.timezone("America/Los_Angeles")
        self.print_optimality = False


    def build_opt_problem(self):
        # build parameters
        self.p_max = cvx.Parameter(self.num_agents)
        self.p_min = cvx.Parameter(self.num_agents)
        self.cost = cvx.Parameter(self.num_agents)
        self.demand = cvx.Parameter()

        # build variables
        self.p = cvx.Variable(self.num_agents)
        self.cleared = cvx.Variable(self.num_agents, boolean=True)

        # build constraints
        constraint = [np.ones(self.num_agents).T * self.p == self.demand]
        for i in range(self.num_agents):
            constraint += [self.p[i] <= self.cleared[i] * self.p_max[i]]
            constraint += [self.cleared[i] * self.p_min[i] <= self.p[i]]

        # build the objective
        objective = cvx.Minimize(self.p.T * self.cost)

        # build objective
        problem = cvx.Problem(objective, constraint)
        return problem

    def reset(self, start_date):
        self.date = start_date

    def step(self, action):
        # assign values to the cvxpy parameters
        self.p_min.value, self.p_max.value, self.cost.value = self.get_bids_actors(action, self.date)
        self.demand.value = self.get_demand(self.date)

        # solve the problem
        self.opt_problem.solve(verbose=False)
        if self.print_optimality or "optimal" not in self.opt_problem.status:
            print(self.opt_problem.status)
            raise(OptimizationException)
        self.date += self.delta_time

        # send result to battery
        try:
            self.p.value[-1], self.cleared.value[-1]
        except TypeError:
            import ipdb;ipdb.set_trace()
        return self.p.value[-1], self.cleared.value[-1]

    def get_demand(self, date):
        load = self.caiso_get_load(start_at=date, end_at=date+self.delta_time)
        load_list = load['load_MW']
        demand = np.mean(load_list)
        return demand


    def get_bids_actors(self, action, date):
        gen = self.caiso_get_generation(start_at=date, end_at=date + self.delta_time)
        if gen.empty:
            raise EmptyDataException
        gen_wind_list = gen[gen['fuel_name'] == 'wind']['gen_MW'].values
        gen_solar_list = gen[gen['fuel_name'] == 'solar']['gen_MW'].values
        gen_other_list = gen[gen['fuel_name'] == 'other']['gen_MW'].values
        p_max = np.array([np.mean(gen_wind_list),
                          np.mean(gen_solar_list),
                          np.mean(gen_other_list),
                          10000 + 10000 * (np.mean(gen_wind_list) + np.mean(gen_solar_list) + np.mean(gen_other_list)),
                          action[0]])
        p_min = p_max.copy()
        p_min[2] = 0
        p_min[3] = 0
        cost = np.array([2, 2, 9, 1000, action[1]])
        print(p_min, p_max, cost)
        return p_min, p_max, cost


    def caiso_get_generation(self, start_at, end_at):
        start_date_aware = self.timezone.localize(start_at)
        end_date_aware = self.timezone.localize(end_at)
        return self.gen_df[(start_date_aware <= self.gen_df["timestamp"]) &
                           (end_date_aware > self.gen_df["timestamp"])]


    def caiso_get_load(self, start_at, end_at):
        start_date_aware = self.timezone.localize(start_at)
        end_date_aware = self.timezone.localize(end_at)
        return self.dem_df[(start_date_aware <= self.dem_df["timestamp"]) &
                           (end_date_aware > self.dem_df["timestamp"])]


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
        penalty = 100
        energy_to_add = self.efficiency_ratio * power
        if self.min_power <= abs(energy_to_add) <= self.max_power:
            next_soe = self.soe + energy_to_add
            if self.min_soe <= next_soe <= self.max_soe:
                self.soe = next_soe
                penalty = 0

        return self.soe, penalty
