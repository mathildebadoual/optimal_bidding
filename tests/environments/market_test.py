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


if __name__ == '__main__':
    unittest.main()
