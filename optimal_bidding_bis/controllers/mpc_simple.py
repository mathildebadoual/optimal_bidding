import sys
import os
import pandas as pd
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))))

from utils.data_postprocess import DataProcessor
from environments.agents import Battery
from environments.energy_market import EnergyMarket

RESULTS_PATH = 'results/mpc_results.csv'


def main():
    # start_time = pd.Timestamp(
    #     year=2018,
    #     month=6,
    #     day=30,
    #     hour=4,
    #     minute=30
    #     )
    # end_time = pd.Timestamp(
    #     year=2018,
    #     month=7,
    #     day=8,
    #     hour=0,
    #     minute=0
    #     )

    start_time = pd.Timestamp(
        year=2018,
        month=6,
        day=2,
        hour=0,
        minute=0
        )
    end_time = pd.Timestamp(
        year=2018,
        month=10,
        day=1,
        hour=0,
        minute=0
        )

    energy_market = EnergyMarket(
        start_time,
        end_time
        )
    battery = Battery()
    end = False
    index = 0

    while not end:
        timestamp = energy_market.get_timestamp()
        print('timestamp: %s' % timestamp)

        soe = battery.get_soe()

        # get the bids from the battery
        battery_bid = battery.bid_mpc(timestamp)

        # run the market dispatch
        bid_cleared, clearing_price, end, demand = energy_market.step(
            battery_bid)

        # get state
        bid_power = bid_cleared.power_signed()

        # update soe of the battery with the cleared power
        battery.step(bid_power)

        save_data(bid_cleared,
                  battery_bid,
                  clearing_price,
                  soe,
                  index,
                  timestamp,
                  demand,
                  )
        index += 1


def save_data(bid_cleared,
              battery_bid,
              clearing_price,
              soe,
              index,
              timestamp,
              demand
              ):
    d = {}
    d['battery_bid_price'] = battery_bid.price()

    if battery_bid.type() == 'gen':
        d['battery_bid_power_gen'] = battery_bid.power_signed()
        d['battery_bid_power_load'] = 0
    else:
        d['battery_bid_power_gen'] = 0
        d['battery_bid_power_load'] = battery_bid.power_signed()

    if bid_cleared.type() == 'gen':
        d['battery_bid_cleared_power_gen'] = bid_cleared.power_signed()
        d['battery_bid_cleared_power_load'] = 0
    else:
        d['battery_bid_cleared_power_gen'] = 0
        d['battery_bid_cleared_power_load'] = bid_cleared.power_signed()

    d['demand'] = demand
    d['clearing_price'] = clearing_price
    d['soe'] = soe
    d['timestamp'] = timestamp

    df = pd.DataFrame(data=d, index=[index])
    with open(RESULTS_PATH, 'a') as f:
        if index == 0:
            df.to_csv(f, header=True)
        else:
            df.to_csv(f, header=False)


if __name__ == '__main__':
    if os.path.exists(RESULTS_PATH):
        os.remove(RESULTS_PATH)
    main()
