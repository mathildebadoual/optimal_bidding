import unittest
from energy_market_model.energy_market_model import Env, MarketModel, StorageSystem
import random
import numpy as np

""" Energy Market Model Tests """


class TestEnv(unittest.TestCase):
    def setUp(self):
        pass

    def test_init(self):
        pass


class TestMarkeTModel(unittest.TestCase):
    def setUp(self):
        self.num_other_agents = 13
        self.grid_parameters = {
                'grid_connexion': np.array([
                        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                'theta_max': 1.05 * np.ones(14),
                'theta_min': 0.95 * np.ones(14),
                'agents_in_grid': np.array([
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]),
                'h': np.array([
                        [0, 0.02640, 0, 0, 0.02190, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0.01870, 0.02460, 0.01700, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0.01730, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0.00640, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                }
        self.index_storage = 13
        self.market_model = MarketModel(self.num_other_agents, self.grid_parameters, self.index_storage)

    def test_init(self):
        pass

    def test_step(self):
        action = [10, 1.2]
        p, cleared = self.market_model.step(action)
        print(p, cleared)
        self.assertEqual(len(cleared), self.num_other_agents)

    def test_reset(self):
        self.market_model.reset()
        self.assertEqual(self.market_model.time, 0)


class TestStorageSystem(unittest.TestCase):
    def setUp(self):
        self.storage_system = StorageSystem()

    def test_init(self):
        pass

    def test_reset(self):
        self.storage_system.reset()
        self.assertEqual(self.storage_system.soe, 0)

    def test_step(self):
        power = random.randint(self.storage_system.min_power,
                               self.storage_system.max_power)
        actual_soe = self.storage_system.step(power)
        self.assertEqual(actual_soe, power * self.storage_system.efficiency_ratio)

        # When it reachs the power limit
        self.storage_system.reset()
        power = self.storage_system.max_power + 2
        actual_soe = self.storage_system.step(power * self.storage_system.efficiency_ratio)
        self.assertEqual(actual_soe, 0)

        # When it reachs the soe limit
        self.storage_system.reset()
        power = self.storage_system.min_soe - 5
        actual_soe = self.storage_system.step(power)
        self.assertEqual(actual_soe, 0)
