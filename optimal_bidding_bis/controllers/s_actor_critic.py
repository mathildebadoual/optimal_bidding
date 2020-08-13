import sys
import os
sys.path.append(os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))))

from optimal_bidding.environments.energy_market import FCASMarket
from optimal_bidding.environments.agents import Bid


def main():
    fcas_market = FCASMarket()
    state = True
    while state:
        action = policy(state)
        state = fcas_market.step(battery_bid=action)


def policy(state):
    return Bid(1, 1)


if __name__ == '__main__':
    main()
