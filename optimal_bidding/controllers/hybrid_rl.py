import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from optimal_bidding.environments.energy_market import FCASMarket
from optimal_bidding.environments.agents import Battery, Bid
from optimal_bidding.utils.nets import ActorNet, CriticNet
import optimal_bidding.utils.data_postprocess as data_utils
import torch


class ActorCritic():
    def __init__(self):
        # hyperparameters
        self._exploration_size = None
        self._actor_step_size = 0.01
        self._critic_step_size = 0.01
        self._discount_factor = 0.95
        self._eligibility_trace_decay_factor = 0.7


        self._fcas_market = FCASMarket()
        self._battery = Battery()
        self._actor_nn = ActorNet()
        self._critic_nn = CriticNet()

        self._eligibility = [0] * len(list(self._critic_nn.parameters()))
        self._delta = None

    def run_simulation(self):
        end = False
        index = 0
        k = 1
        # Initialize to 0 (only affects first state)
        energy_cleared_price = None
        fcas_clearing_price = 0
        while not end:
            timestamp = self._fcas_market.get_timestamp()
            print('timestamp: %s' % timestamp)

            # create the state:
            soe = self._battery.get_soe()
            step_of_day = self._get_step_of_day(timestamp)

            # energy_cleared_price will hold previous clearing price unless it is the first timestamp
            if energy_cleared_price is None:
                energy_cleared_price = data_utils.get_energy_price(timestamp)
            prev_energy_cleared_price = energy_cleared_price
            energy_cleared_price = data_utils.get_energy_price(timestamp)
            prev_fcas_clearing_price = fcas_clearing_price
            raise_demand = data_utils.get_raise_demand(timestamp)
            # TODO finish state definition
            state = torch.tensor(np.array([step_of_day,
                                          soe,
                                          prev_energy_cleared_price,
                                          energy_cleared_price,
                                          prev_fcas_clearing_price,
                                          raise_demand]))

            # compute the action = [p_raise, c_raise, p_energy]
            action_supervisor, action_actor, action_exploration, action_composite = self._compute_action(state, timestamp, k)
            energy_cleared_price = data_utils.get_energy_price(timestamp)

            bid_fcas, bid_energy = self._transform_to_bid(
                action_composite, energy_cleared_price)

            # run the market dispatch
            fcas_bid_cleared, fcas_clearing_price, end = self._fcas_market.step(
                bid_fcas)

            # update soe of the battery with the cleared power
            self._battery.step(fcas_bid_cleared.power_signed(),
                               bid_energy.power_signed())

            reward = self._compute_reward(bid_fcas, bid_energy,
                                          energy_cleared_price,
                                          fcas_bid_cleared)

            next_soe = self._battery.get_soe()

            next_timestamp = timestamp + pd.Timedelta('30 min')
            next_step_of_day = self._get_step_of_day(next_timestamp)
            next_energy_cleared_price = data_utils.get_energy_price(next_timestamp)
            next_raise_demand = data_utils.get_raise_demand(next_timestamp)
            next_state = torch.tensor([next_step_of_day,
                                       next_soe,
                                       energy_cleared_price,
                                       next_energy_cleared_price,
                                       fcas_clearing_price,
                                       next_raise_demand])

            # update eligibility and delta
            current_state_value = self._critic_nn(state.float())
            next_state_value = self._critic_nn(next_state.float())
            self._delta = reward + self._discount_factor * next_state_value - current_state_value

            # update neural nets
            self._update_critic(current_state_value)
            self._update_actor(action_supervisor, action_actor, action_exploration, k)

            index += 1

    def _update_critic(self, current_state_value):

        self._critic_nn.zero_grad()
        current_state_value.backward()
        i = 0
        for f in self._critic_nn.parameters():
            # update eligibilities
            self._eligibility[i] = self._discount_factor * self._eligibility_trace_decay_factor * self._eligibility[i] + f.grad.data
            # update weights. not sure whether the minus sign should be there.
            f.data.sub_(- self._critic_step_size * self._delta * self._eligibility[i])
            i += 1


    def _update_actor(self, action_supervisor, action_actor, action_exploration, k):
        self._actor_nn.zero_grad()
        action_actor.backward(torch.ones(1,3))
        print(action_actor)
        print(action_supervisor)
        print(action_exploration)
        for f in self._actor_nn.parameters():
            # update weights. not sure whether the minus sign should be there.
            f.data.sub_(- self._actor_step_size * ((1-k) * self._delta * action_exploration + k * (action_supervisor - action_actor)) * f.grad.data)



    def _transform_to_bid(self, action, energy_cleared_price):
        action = action[0].data.numpy()
        if action[1] < 0:
            action[1] = 0
        if action[0] < 0:
            action[0] = 0
        bid_fcas = Bid(action[0], action[1], bid_type='gen')
        if action[2] >= 0:
            bid_energy = Bid(action[2], energy_cleared_price, bid_type='load')
        else:
            bid_energy = Bid(action[2], energy_cleared_price, bid_type='gen')
        return bid_fcas, bid_energy


    def _compute_action(self, state, timestamp, k):
        # timestamp = datetime.fromtimestamp(timestamp)
        bid_fcas_mpc, bid_energy_mpc = self._battery.bid_mpc(timestamp)
        action_supervisor = torch.tensor([
            bid_fcas_mpc.power_signed(),
            bid_fcas_mpc.price(),
            bid_energy_mpc.power_signed()
        ])
        action_actor = self._actor_nn(state.float())
        action_exploration = torch.randn(1,3)
        return action_supervisor, action_actor, action_exploration, k * action_supervisor + (1 - k) * (action_actor + action_exploration)

    def _compute_reward(self, bid_fcas, bid_energy, energy_cleared_price,
                        fcas_bid_cleared):
        # assume the markets are pay-as-bid
        # assume the energy market always clears your bid
        energy_cleared_power = bid_energy.power_signed()
        # energy_bid_price = bid_energy.price()

        fcas_cleared_price = bid_fcas.price()
        fcas_bid_power = bid_fcas.power_signed()
        fcas_cleared_power = fcas_bid_cleared.power_signed()

        # bare bones reward function
        reward = - energy_cleared_power * energy_cleared_price +\
                0.9 * fcas_cleared_power * fcas_cleared_price

        soe = self._battery.get_soe()
        total_capacity = self._battery._total_capacity
        max_power = self._battery._max_power
        max_ramp = self._battery._max_ramp

        new_energy = soe + self._battery._efficiency * energy_cleared_power +\
                self._battery._ratio_fcast * fcas_cleared_power

        # weight the constraints by how 'much' the constraint
        # is violated multiplied by some scalar. this can be changed.
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
        if -energy_cleared_power > max_power:
            reward -= penalty * (-energy_cleared_power - max_power)

        return reward

    def _get_step_of_day(self, timestamp, timestep_min=30):
        return timestamp.hour * 60/timestep_min + timestamp.minute / timestep_min

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


def main():
    actor_critic = ActorCritic()
    actor_critic.run_simulation()


if __name__ == '__main__':
    main()
