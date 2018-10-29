import unittest
from models.energy_market_model import Env, MarketModel, StorageSystem
import random
import numpy as np
import datetime


""" Energy Market Model Tests """


class TestEnv(unittest.TestCase):
    def setUp(self):
        pass

    def test_init(self):
        pass


class TestMarketModel(unittest.TestCase):
    def setUp(self):
        self.num_agents = 4
        start_date = datetime.datetime(2018, 10, 24, 7, 25)
        self.market_model = MarketModel(self.num_agents, start_date)

    def test_init(self):
        pass

    def test_step(self):
        action = [10000, 2]
        p, cleared = self.market_model.step(action)
        print(p, cleared)
        self.assertEqual(len(cleared), self.num_agents)

    def test_reset(self):
        start_date = datetime.datetime(2018, 10, 24, 7, 00)
        self.market_model.reset(start_date)
        self.assertEqual(self.market_model.date, start_date)

    def test_get_demand(self):
        date = self.market_model.date
        demand = self.market_model.get_demand(date)
        self.assertIsInstance(demand, float)


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
        actual_soe, penalty = self.storage_system.step(power)
        self.assertEqual(actual_soe, power * self.storage_system.efficiency_ratio)

        # When it reachs the power limit
        self.storage_system.reset()
        power = self.storage_system.max_power + 2
        actual_soe, penalty = self.storage_system.step(power * self.storage_system.efficiency_ratio)
        self.assertEqual(actual_soe, 0)

        # When it reachs the soe limit
        self.storage_system.reset()
        power = self.storage_system.min_soe - 5
        actual_soe, penalty = self.storage_system.step(power)
        self.assertEqual(actual_soe, 0)
