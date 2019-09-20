import unittest
import pandas as pd
import numpy as np

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import optimal_bidding.utils.data_postprocess as data_utils


class TestGetFunctions(unittest.TestCase):
    def setUp(self):
        self._timestamp = pd.Timestamp(
                year=2018,
                month=6,
                day=3,
                hour=5,
                minute=0,
                )

    def test_get_energy_demand(self):
        demand = data_utils.get_energy_demand(self._timestamp)
        self.assertEqual(demand, 1168.02)

    def test_get_energy_prices(self):
        price = data_utils.get_energy_price(self._timestamp)
        self.assertEqual(price, 80.95)

    def test_get_energy_price_day_ahead(self):
        price = data_utils.get_energy_price_day_ahead(self._timestamp,
                                                      horizon=2)
        self.assertTrue(np.array_equal(price, [80.95, 84.29]))

    def test_get_energy_demand_day_ahead(self):
        demand = data_utils.get_energy_demand_day_ahead(self._timestamp,
                                                        horizon=2)
        self.assertTrue(np.array_equal(demand, [1168.02, 1163.29]))

    def test_get_raise_demand(self):
        demand = data_utils.get_raise_demand(self._timestamp)
        self.assertEqual(demand, 41.0)

    def test_get_raise_prices(self):
        price = data_utils.get_raise_price(self._timestamp)
        self.assertEqual(price, 28.49)

    def test_get_raise_price_day_ahead(self):
        price = data_utils.get_raise_price_day_ahead(self._timestamp,
                                                     horizon=2)
        self.assertTrue(np.allclose(price, [28.49, 26.44956], rtol=1e-2))

    def test_get_raise_demand_day_ahead(self):
        demand = data_utils.get_raise_demand_day_ahead(self._timestamp,
                                                       horizon=2)
        self.assertTrue(np.array_equal(demand, [41.0, 41.0]))

    def test_get_low_demand(self):
        demand = data_utils.get_low_demand(self._timestamp)
        self.assertEqual(demand, 40.0)

    def test_get_low_prices(self):
        price = data_utils.get_low_price(self._timestamp)
        self.assertEqual(price, 0.2)

    def test_get_low_price_day_ahead(self):
        price = data_utils.get_low_price_day_ahead(self._timestamp, horizon=2)
        self.assertTrue(np.array_equal(price, [0.2, 0.44]))

    def test_get_low_demand_day_ahead(self):
        demand = data_utils.get_low_demand_day_ahead(self._timestamp,
                                                     horizon=2)
        self.assertTrue(np.array_equal(demand, [40.0, 40.0]))


if __name__ == '__main__':
    unittest.main()
