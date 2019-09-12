import sys
import os
sys.path.append(os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))))

from optimal_bidding.environments.energy_market import FCASMarket


def main():
    fcas_market = FCASMarket()
    while fcas_market.step():
        print('ok')


if __name__ == '__main__':
    main()
