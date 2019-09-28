import sys
import os
import pandas as pd
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from optimal_bidding.environments.energy_market import FCASMarket
from optimal_bidding.environments.agents import Battery
from optimal_bidding.utils.data_postprocess import DataProcessor


def main():
    fcas_market = FCASMarket()
    battery = Battery()
    end = False
    index = 0
    last_day_of_data = pd.Timestamp(
        year=2018,
        month=11,
        day=1,
        hour=0,
        minute=0,
    )
    filename = 'FiveMonths2018_30min.csv'
    data_utils = DataProcessor(last_day_of_data, filename)
    while not end:
        timestamp = fcas_market.get_timestamp()
        print('timestamp: %s' % timestamp)

        soe = battery.get_soe()

        # get the bids from the battery
        battery_bid_fcas, battery_bid_energy = battery.bid_mpc(timestamp)

        # run the market dispatch
        bid_fcas_cleared, fcas_clearing_price, end = fcas_market.step(battery_bid_fcas)

        # get state
        fcas_cleared_power = bid_fcas_cleared.power()

        # update soe of the battery with the cleared power
        battery.step(fcas_cleared_power, battery_bid_energy.power_signed())

        energy_price = data_utils.get_energy_price(timestamp)
        raise_demand = data_utils.get_raise_demand(timestamp)
        raise_price = data_utils.get_raise_price(timestamp)

        save_data(battery_bid_fcas, battery_bid_energy, fcas_cleared_power,
                  fcas_clearing_price, soe, index, timestamp, energy_price,
                  raise_demand, raise_price)
        index += 1


def save_data(battery_bid_fcas, battery_bid_energy, fcas_cleared_power,
              fcas_clearing_price, soe, index, timestamp, energy_price,
              raise_demand, raise_price):
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
    d['raise_demand'] = raise_demand
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
