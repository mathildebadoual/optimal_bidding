import unittest

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

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


class TestPVAgent(unittest.TestCase):
    def setUp(self)i:
        self._pv_agent = energy_market.PVAgent()

    def test_bid(self):
        self._pv_agent.bid()

if __name__ == '__main__':
    unittest.main()
