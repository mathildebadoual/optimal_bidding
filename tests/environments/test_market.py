import unittest

import sys
import os
sys.path.append(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from optimal_bidding.environments import energy_market


class TestBid(unittest.TestCase):
    def setUp(self):
        self._bid = energy_market.Bid()

    def test_power(self):
        self.assertEqual(self._bid.power(), None)


class TestAgent(unittest.TestCase):
    def setUp(self):
        self._agent = energy_market.Agent()

    def test_bid(self):
        self.assertRaises(NotImplementedError, self._agent.bid())


class TestFCASMarket(unittest.TestCase):
    def setUp(self):
        self._market = energy_market.FCASMarket()

    def test_create_agents(self):
        pass

    def test_compute_dispatch(self):
        pass


if __name__ == '__main__':
    unittest.main()
