import unittest
import pandas as pd

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from optimal_bidding.utils.data_postprocess import TransitionMap, get_demand, get_energy_price


class TestGetFunctions(unittest.TestCase):
    def setUp(self):
        self._timestamp = pd.Timestamp(
                year=2018,
                month=6,
                day=3,
                hour=5,
                minute=0,
                )

    def test_get_demand(self):
        demand = get_demand(self._timestamp)
        self.assertEqual(demand, 1168.02)

    def test_get_energy_prices(self):
        price = get_energy_price(self._timestamp)
        self.assertEqual(price, 80.95)


if __name__ == '__main__':
    unittest.main()
