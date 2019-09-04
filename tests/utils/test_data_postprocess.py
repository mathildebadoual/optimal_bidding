import unittest

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from optimal_bidding.utils.data_postprocess import TransitionMap


class TestTransitionMap(unittest.TestCase):
    def setUp(self):
        self.transition_map = TransitionMap('PV')

    def test_get_next_state(self):
        current_state = 0
        hour = 0
        self.transition_map.get_next_state(current_state, hour)


if __name__ == '__main__':
    unittest.main()
