import sys
import os
import pandas as pd
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from optimal_bidding.environments.energy_market import FCASMarket
from optimal_bidding.environments.agents import Battery
from optimal_bidding.utils.nets import ActorNet, CriticNet
import optimal_bidding.utils.data_postprocess as data_utils


def main():

    # hyperparameters
    exploration_size = None  # Covariance matrix of the normal distribution used to explore
    actor_step_size = 0.01
    critic_step_size = 0.01
    discount_factor = 0.95
    eligibility_trace_decay_factor = 0.7


def reward_function(battery, bid_fcas, bid_energy, energy_cleared_price, fcas_cleared_power):
    # assume the markets are pay-as-bid
    # assume the energy market always clears your bid
    energy_cleared_power = bid_energy.power_signed()
    energy_bid_price = bid_energy.price()

    fcas_cleared_price = bid_fcas.price()
    fcas_bid_power = bid_fcas.power_signed()

    # bare bones reward function
    reward = -energy_cleared_power * energy_cleared_price + 0.9 * fcas_cleared_power * fcas_cleared_price

    soe = battery.get_soe()
    total_capacity = battery._total_capacity
    max_power = battery._max_power
    max_ramp = battery._max_ramp

    new_energy = soe + battery._efficiency * energy_cleared_power + \
            battery._ratio_fcast * fcas_cleared_power

    # weight the constraints by how 'much' the constraint is violated multiplied by some scalar. this can be changed.
    # only punish if bounds on capacity, power, or ramp are violated.
    penalty = 50

    if new_energy > total_capacity:
        reward -= penalty * (new_energy - total_capacity)
    if new_energy < 0:
        reward -= penalty * (-new_energy)
    if -fcas_bid_power > max_ramp:
        reward -= penalty * fcas_bid_power
    # penalize "low" fcas bids
    if fcas_bid_power > 0:
        reward -= penalty * fcas_bid_power
    if -fcas_bid_power > max_ramp:
        reward -= penalty * (-fcas_bid_power - max_ramp)
    if -energy_cleared_power  > max_power:
        reward -= penalty * (-energy_cleared_power - max_power)
    return reward


def create_actor_critic(n_input, n_hidden, n_output):
    ## initialize tensor for inputs, and outputs
    x = torch.randn((1, n_input))
    y = torch.randn((1, n_output))

    ## initialize tensor variables for weights
    w1 = torch.randn(n_input, n_hidden)
    w2 = torch.randn(n_hidden, n_output)

    ## initialize tensor variables for bias terms
    b1 = torch.randn((1, n_hidden))
    b2 = torch.randn((1, n_output))

    ## sigmoid activation function using pytorch
    def sigmoid_activation(z):
        return 1 / (1 + torch.exp(-z))

    ## activation of hidden layer
    z1 = torch.mm(x, w1) + b1
    a1 = sigmoid_activation(z1)

    ## activation (output) of final layer
    z2 = torch.mm(a1, w2) + b2
    model = sigmoid_activation(z2)

    return model


def get_gradient(model, input_values):
    output = model(input_values)
    output.backward()
    x.grad



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
