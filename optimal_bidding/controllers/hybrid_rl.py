import sys
import os
import pandas as pd
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from optimal_bidding.environments.energy_market import FCASMarket
from optimal_bidding.environments.agents import Battery
import optimal_bidding.utils.data_postprocess as data_utils


def main():

    # hyperparameters
    exploration_size = None  # Covariance matrix of the normal distribution used to explore
    actor_step_size = 0.01
    critic_step_size = 0.01
    discount_factor = 0.95
    eligibility_trace_decay_factor = 0.7


def run_simulation():
    """The learning is online so only one function is running
    the all simulation + learning.
    """
    fcas_market = FCASMarket()
    battery = Battery()
    end = False
    index = 0
    while not end:
        timestamp = fcas_market.get_timestamp()
        print('timestamp: %s' % timestamp)

        soe = battery.get_soe()

        # get the bids from the battery
        battery_bid_fcas_mpc, battery_bid_energy_mpc = battery.bid_mpc(timestamp)

        # run the market dispatch
        state = fcas_market.step(battery_bid_fcas)

        # get state
        fcas_cleared_power = state[0]
        fcas_clearing_price = state[1]
        end = state[2]

        # update soe of the battery with the cleared power
        battery.step(fcas_cleared_power, battery_bid_energy.power_signed())

        energy_price = data_utils.get_energy_price(timestamp)
        low_price = data_utils.get_low_price(timestamp)
        raise_price = data_utils.get_raise_price(timestamp)

        save_data(battery_bid_fcas, battery_bid_energy, fcas_cleared_power,
                  fcas_clearing_price, soe, index, timestamp, energy_price,
                  low_price, raise_price)
        index += 1



def save_data(battery_bid_fcas, battery_bid_energy, fcas_cleared_power,
              fcas_clearing_price, soe, index, timestamp, energy_price,
              low_price, raise_price):
    """This function is just to save the data in a csv. To be changed as needed!
    """
    d = {}
    d['battery_bid_fcas_power'] = battery_bid_fcas.power()
    d['battery_bid_fcas_price'] = battery_bid_fcas.price()
    d['battery_bid_fcas_type'] = battery_bid_fcas.type()

    if battery_bid_energy.type() == 'gen':
        d['battery_bid_energy_power_gen'] = battery_bid_energy.power_signed()
        d['battery_bid_energy_power_load'] = 0
    else:
        d['battery_bid_energy_power_gen'] = 0
        d['battery_bid_energy_power_load'] = battery_bid_energy.power_signed()

    d['battery_bid_energy_price'] = battery_bid_energy.price()
    d['battery_bid_energy_type'] = battery_bid_energy.type()

    d['fcas_clearing_price'] = fcas_clearing_price
    d['energy_price'] = energy_price
    d['low_price'] = low_price
    d['raise_price'] = raise_price
    d['soe'] = soe
    d['timestamp'] = timestamp

    df = pd.DataFrame(data=d, index=[index])
    with open('mpc_results.csv', 'a') as f:
        if index == 0:
            df.to_csv(f, header=True)
        else:
            df.to_csv(f, header=False)


if __name__ == '__main__':
    main()
