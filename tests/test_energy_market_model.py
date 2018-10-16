import unittest
from energy_market_model.energy_market_model import Env, MarketModel, StorageSystem
import random


""" Energy Market Model Tests """


class TestEnv(unittest.TestCase):
    def setUp(self):
        self.num_other_agents = 5
        self.env = Env(self.num_other_agents)

    def test_init(self):
        pass


class TestMarkeTModel(unittest.TestCase):
    def setUp(self):
        self.num_other_agents = 5
        self.market_model = MarketModel(self.num_other_agents)

    def test_init(self):
        pass

    def test_step(self):
        action = (3, 10)
        cleared_bids = self.market_model.step(action)
        self.assertEqual(len(cleared_bids), self.num_other_agents + 1)

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
