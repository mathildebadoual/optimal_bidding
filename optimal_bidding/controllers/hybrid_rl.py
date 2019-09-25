import sys
import os
import pandas as pd
import numpy as np

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from optimal_bidding.environments.energy_market import FCASMarket
from optimal_bidding.environments.agents import Battery, Bid
from optimal_bidding.utils.nets import ActorNet, CriticNet, Network, Layer
import optimal_bidding.utils.data_postprocess as data_utils
import torch


class ActorCritic():
    def __init__(self):
        # hyperparameters
        self._sig = 0.1

        self._actor_step_size = 0.1
        self._critic_step_size = 0.01
        self._discount_factor = 0.9
        self._e_tdf = 0.1

        torch.manual_seed(0)
        self._fcas_market = FCASMarket()
        self._battery = Battery()

        l1w = np.random.normal(0, self._sig, (20, 6))
        l1b = np.random.normal(0, self._sig, 20)

        l2w = np.random.normal(0, self._sig, (10, 20))
        l2b = np.random.normal(0, self._sig, 10)

        l3w = np.random.normal(0, self._sig, (8, 10))
        l3b = np.random.normal(0, self._sig, 8)

        l4w = np.random.normal(0, self._sig, (1, 8))
        l4b = np.random.normal(0, self._sig, 1)

        l5w = np.random.normal(0, self._sig, (3, 8))
        l5b = np.random.normal(0, self._sig, 3)

        self._actor_nn = Network(
            [Layer(l1w, l1b, False),
             Layer(l2w, l2b),
             Layer(l3w, l3b),
             Layer(l5w, l5b, False)])
        self._critic_nn = Network(
            [Layer(l1w, l1b, False),
             Layer(l2w, l2b),
             Layer(l3w, l3b),
             Layer(l4w, l4b, False)])

        self._e = [0] * len(self._critic_nn.layers)
        self._delta = None

    def run_simulation(self):
        end = False
        index = 0
        k = 0
        # Initialize to 0 (only affects first state)
        energy_cleared_price = 0
        fcas_clearing_price = 0
        while not end:
            timestamp = self._fcas_market.get_timestamp()
            print('---------- timestamp: %s ----------' % timestamp)

            # create the state:
            soe = self._battery.get_soe()
            step_of_day = self._get_step_of_day(timestamp)

            # energy_cleared_price will hold previous clearing price
            # unless it is the first timestamp
            if energy_cleared_price is None:
                energy_cleared_price = data_utils.get_energy_price(timestamp)
            prev_energy_cleared_price = energy_cleared_price
            energy_cleared_price = data_utils.get_energy_price(timestamp)
            prev_fcas_clearing_price = fcas_clearing_price
            raise_demand = data_utils.get_raise_demand(timestamp)
            state = np.array([
                step_of_day, soe, prev_energy_cleared_price,
                energy_cleared_price, prev_fcas_clearing_price, raise_demand
            ])

            # compute the action = [p_raise, c_raise, p_energy]
            action_supervisor, action_actor, action_exploration, action_composite = self._compute_action(
                state, timestamp, k)
            energy_cleared_price = data_utils.get_energy_price(timestamp)

            bid_fcas, bid_energy = self._transform_to_bid(
                action_composite, energy_cleared_price)

            bid_fcas, bid_energy, reward_shield = self._shield(
                bid_fcas, bid_energy, action_supervisor)

            print('bid power: %s' % bid_fcas.power_signed())
            print('bid price: %s' % bid_fcas.price())
            print('reward shield: %s' % reward_shield)

            # run the market dispatch
            bid_fcas_cleared, fcas_clearing_price, end = self._fcas_market.step(
                bid_fcas)

            # update soe of the battery with the cleared power
            self._battery.step(bid_fcas_cleared.power_signed(),
                               bid_energy.power_signed())

            reward = self._compute_reward(bid_fcas, bid_energy,
                                          energy_cleared_price,
                                          bid_fcas_cleared,
                                          reward_shield)
            next_soe = self._battery.get_soe()

            next_timestamp = timestamp + pd.Timedelta('30 min')
            next_step_of_day = self._get_step_of_day(next_timestamp)
            next_energy_cleared_price = data_utils.get_energy_price(next_timestamp)
            next_raise_demand = data_utils.get_raise_demand(next_timestamp)
            next_state = np.array([
                next_step_of_day, next_soe, energy_cleared_price,
                next_energy_cleared_price, fcas_clearing_price,
                next_raise_demand
            ])

            print(action_actor)

            # update eligibility and delta
            current_state_value = self._critic_nn.compute_value(state)
            next_state_value = self._critic_nn.compute_value(next_state)
            self._delta = reward + self._discount_factor * next_state_value -\
                    current_state_value

            bid_fcas_actor_power = abs(action_actor[0])
            bid_fcas_actor_price = action_actor[1]
            bid_energy_actor_power = action_actor[2]

            # update neural nets
            self._update_critic(state)
            self._update_actor(action_supervisor, action_actor, action_exploration, k, state)

            save_data(bid_fcas, bid_energy, bid_fcas_cleared,
                      fcas_clearing_price, soe, index, timestamp,
                      energy_cleared_price, reward, current_state_value,
                      next_state_value, raise_demand, bid_fcas_actor_power,
                      bid_fcas_actor_price, bid_energy_actor_power, k,
                      self._delta)

            index += 1
            if index % 100 == 0 and k < 1:
                k += 0.1
                print(k)

    def _update_critic(self, state):
        grad_critic = self._critic_nn.compute_value_and_derivative_w(state)
        for i, l in enumerate(self._critic_nn.layers):
            self._e[i] = self._discount_factor * self._e_tdf * self._e[
                i] + grad_critic[i]
            l.weights_matrix += -self._critic_step_size * self._delta * self._e[
                i]

    def _update_actor(self, action_supervisor, action_actor,
                      action_exploration, k, state):
        action_vector = (k * self._delta * action_exploration + (1 - k) *
                         (action_supervisor - action_actor))[0]
        grad_action = self._actor_nn.compute_value_and_derivative_w(state)
        update = self._actor_step_size * np.dot(action_vector, grad_action)

    def _transform_to_bid(self, action, energy_cleared_price):
        action = action[0]
        bid_fcas = Bid(-action[0], action[1], bid_type='gen')
        if action[2] >= 0:
            bid_energy = Bid(action[2], energy_cleared_price, bid_type='load')
        else:
            bid_energy = Bid(-action[2], energy_cleared_price, bid_type='gen')
        return bid_fcas, bid_energy

    def _shield(self, bid_fcas, bid_energy, action_supervisor):
        reward = 0
        if bid_fcas.power() < 0:
            bid_fcas = Bid(-action_supervisor[0], action_supervisor[1], bid_type='gen')
            reward -= 100 * abs(bid_fcas.power())
        return bid_fcas, bid_energy, reward

    def _compute_action(self, state, timestamp, k):
        # timestamp = datetime.fromtimestamp(timestamp)
        bid_fcas_mpc, bid_energy_mpc = self._battery.bid_mpc(timestamp)
        action_supervisor = np.array([
            bid_fcas_mpc.power_signed(),
            bid_fcas_mpc.price(),
            bid_energy_mpc.power_signed()
        ])
        action_actor = self._actor_nn.compute_value(state)
        action_exploration = np.random.normal(0, self._sig, (1, 3))
        return action_supervisor, action_actor, action_exploration, (1 - k) * action_supervisor + k  * (action_actor + action_exploration)

    def _compute_reward(self, bid_fcas, bid_energy, energy_cleared_price,
                        bid_fcas_cleared, reward_shield):
        # assume the markets are pay-as-bid
        # assume the energy market always clears your bid
        bid_energy_cleared_power = bid_energy.power_signed()
        bid_energy_cleared_price = bid_energy.price()

        bid_fcas_price = bid_fcas.price()
        bid_fcas_power = bid_fcas.power_signed()
        bid_fcas_cleared_power = bid_fcas_cleared.power()
        bid_fcas_cleared_price = bid_fcas_cleared.price()

        # bare bones reward function
        reward = - (bid_energy_cleared_power + abs(bid_fcas_cleared_power)) * bid_energy_cleared_price + \
                bid_fcas_cleared_power * bid_fcas_cleared_price

        soe = self._battery.get_soe()
        total_capacity = self._battery._total_capacity
        max_power = self._battery._max_power
        max_ramp = self._battery._max_ramp

        new_energy = soe + self._battery._efficiency * bid_energy_cleared_power +\
                self._battery._ratio_fcast * bid_fcas_cleared_power

        # weight the constraints by how 'much' the constraint
        # is violated multiplied by some scalar. this can be changed.
        # only punish if bounds on capacity, power, or ramp are violated.
        penalty = 100

        if new_energy > total_capacity:
            reward -= penalty * (new_energy - total_capacity)
        if new_energy < 0:
            reward -= penalty * (-new_energy)
        if -bid_fcas_power > max_ramp:
            reward -= penalty * bid_fcas_power

        # penalize "low" fcas bids
        if bid_fcas_power > 0:
            reward -= penalty * bid_fcas_power
        if -bid_fcas_power > max_ramp:
            reward -= penalty * abs(-bid_fcas_power - max_ramp)
        if -bid_energy_cleared_power > max_power:
            reward -= penalty * abs(-bid_energy_cleared_power - max_power)

        if bid_fcas_price < 0:
            reward -= penalty * abs(bid_fcas_price)

        return reward

    def _get_step_of_day(self, timestamp, timestep_min=30):
        return timestamp.hour * 60 / timestep_min + \
                timestamp.minute / timestep_min


def save_data(bid_fcas, bid_energy, bid_fcas_cleared,
              fcas_clearing_price, soe, index, timestamp, energy_price, reward,
              current_state_value, next_state_value, raise_demand,
              bid_fcas_actor_power, bid_fcas_actor_price,
              bid_energy_actor_power, k, delta):
    d = {}
    d['bid_fcas_power'] = bid_fcas.power()
    d['bid_fcas_price'] = bid_fcas.price()
    d['bid_fcas_type'] = bid_fcas.type()

    d['bid_fcas_actor_power'] = bid_fcas_actor_power
    d['bid_fcas_actor_price'] = bid_fcas_actor_price
    d['bid_energy_actor_power'] = bid_energy_actor_power

    if bid_energy.type() == 'gen':
        d['bid_energy_power_gen'] = bid_energy.power_signed()
        d['bid_energy_power_load'] = 0
    else:
        d['bid_energy_power_gen'] = 0
        d['bid_energy_power_load'] = bid_energy.power_signed()

    d['bid_energy_price'] = bid_energy.price()
    d['bid_energy_type'] = bid_energy.type()
    d['bid_fcas_cleared_power'] = bid_fcas_cleared.power_signed()
    d['fcas_clearing_price'] = fcas_clearing_price
    d['energy_price'] = energy_price
    d['reward'] = reward
    d['current_state_value'] = current_state_value.item()
    d['next_state_value'] = next_state_value.item()
    d['fcas_demand'] = raise_demand
    d['soe'] = soe
    d['timestamp'] = timestamp
    d['k'] = k
    d['delta'] = delta.data.numpy()

    df = pd.DataFrame(data=d, index=[index])
    with open('hybrid_rl_results.csv', 'a') as f:
        if index == 0:
            df.to_csv(f, header=True)
        else:
            df.to_csv(f, header=False)


def main():
    actor_critic = ActorCritic()
    actor_critic.run_simulation()


if __name__ == '__main__':
    main()
