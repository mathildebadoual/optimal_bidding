import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from optimal_bidding.environments.energy_market import FCASMarket
from optimal_bidding.environments.agents import Battery, Bid
from optimal_bidding.utils.nets import ActorNet, CriticNet
import optimal_bidding.utils.data_postprocess as data_utils


class ActorCritic():
    def __init__(self):
        # hyperparameters
        self._exploration_size = None

        self._actor_step_size = 0.01
        self._critic_step_size = 0.01
        self._discount_factor = 0.95

        torch.manual_seed(1)
        self._fcas_market = FCASMarket()
        self._battery = Battery()
        self._actor_nn = ActorNet()
        self._critic_nn = CriticNet()
        self._criterion = torch.nn.MSELoss()

        self._optimizer_actor = optim.Adagrad(self._actor_nn.parameters(),
                                              lr=0.01)
        self._optimizer_critic = optim.SGD(self._critic_nn.parameters(),
                                           lr=0.1)

    def run_simulation(self):
        end = False
        index = 0
        k = 0
        # Initialize to 0 (only affects first state)
        en_cleared_price = 0
        fcas_clearing_price = 0
        while not end:
            timestamp = self._fcas_market.get_timestamp()
            print('---------- timestamp: %s ----------' % timestamp)

            # create the state:
            soe = self._battery.get_soe()
            step_of_day = self._get_step_of_day(timestamp)

            # en_cleared_price will hold previous clearing price
            # unless it is the first timestamp
            if en_cleared_price is None:
                en_cleared_price = data_utils.get_energy_price(timestamp)
            prev_en_cleared_price = en_cleared_price
            en_cleared_price = data_utils.get_energy_price(timestamp)
            prev_fcas_clearing_price = fcas_clearing_price
            raise_demand = data_utils.get_raise_demand(timestamp)
            state = torch.tensor([
                    step_of_day, soe, prev_en_cleared_price,
                    en_cleared_price, prev_fcas_clearing_price,
                    raise_demand
                ])

            # compute the action = [p_raise, c_raise, p_en]
            a_s, a_a, a_e, action_composite = self._compute_action(
                state, timestamp, k)

            # update the actor
            self._update_actor_supervised(a_s, a_a)

            en_cleared_price = data_utils.get_energy_price(timestamp)

            b_fcas, b_en = self._transform_to_bid(
                action_composite, en_cleared_price)

            b_fcas, b_en, r_shield = self._shield(b_fcas, b_en, a_s)

            print('action actor: %s' % a_a)
            print('action supervisor: %s' % a_s)

            # run the market dispatch
            b_fcas_cleared, fcas_clearing_price, end = self._fcas_market.step(
                b_fcas)

            # update soe of the battery with the cleared power
            self._battery.step(b_fcas_cleared.power_signed(),
                               b_en.power_signed())

            r = self._compute_reward(b_fcas, b_en, en_cleared_price,
                                     b_fcas_cleared, r_shield)
            next_soe = self._battery.get_soe()

            next_timestamp = timestamp + pd.Timedelta('30 min')
            next_step_of_day = self._get_step_of_day(next_timestamp)
            next_en_cleared_price = data_utils.get_energy_price(
                    next_timestamp)
            next_raise_demand = data_utils.get_raise_demand(next_timestamp)
            next_state = torch.tensor([
                next_step_of_day, next_soe, en_cleared_price,
                next_en_cleared_price, fcas_clearing_price,
                next_raise_demand
            ])

            # update eligibility
            value = self._critic_nn(state.float())
            next_value = self._critic_nn(next_state.float())
            eligibility = r + self._discount_factor * next_value - value

            # update neural nets
            self._update_critic(eligibility)
            value = self._critic_nn(state.float())
            self._update_actor(value)

            # save data
            a_a_numpy = a_a.data.numpy()
            b_fcas_actor_power = abs(a_a_numpy[0])
            b_fcas_actor_price = a_a_numpy[1]
            b_en_actor_power = a_a_numpy[2]
            save_data(b_fcas, b_en, b_fcas_cleared,
                      fcas_clearing_price, soe, index, timestamp,
                      en_cleared_price, r, value,
                      next_value, raise_demand,
                      b_fcas_actor_power, b_fcas_actor_price,
                      b_en_actor_power, k, eligibility)

            index += 1
            if index > 1000:
                if index % 10 == 0 and k < 1:
                    k += 0.01
                    print(k)

    def _update_critic(self, eligibility):
        loss = self._criterion(eligibility, torch.zeros(len(eligibility)))
        loss.backward()
        self._optimizer_critic.step()
        self._optimizer_critic.zero_grad()

    def _update_actor_supervised(self, a_s, a_a):
        loss = self._criterion(a_a, a_s)
        loss.backward()
        self._optimizer_actor.step()
        self._optimizer_actor.zero_grad()

    def _update_actor(self, value):
        loss = self._criterion(value, torch.zeros(len(value)))
        loss.backward()
        for f in self._actor_nn.parameters():
            f.data.sub(- f.grad.data * self._actor_step_size)
        self._optimizer_actor.zero_grad()

    def _transform_to_bid(self, action, en_cleared_price):
        action = action[0].data.numpy()
        b_fcas = Bid(-action[0], action[1], bid_type='gen')
        if action[2] >= 0:
            b_en = Bid(action[2], en_cleared_price, bid_type='load')
        else:
            b_en = Bid(-action[2], en_cleared_price, bid_type='gen')
        return b_fcas, b_en

    def _shield(self, b_fcas, b_en, a_s):
        r = 0
        if b_fcas.power() < 0:
            b_fcas = Bid(-a_s[0], a_s[1], bid_type='gen')
            r -= 100 * abs(b_fcas.power())
        return b_fcas, b_en, r

    def _compute_action(self, state, timestamp, k):
        # timestamp = datetime.fromtimestamp(timestamp)
        b_fcas_mpc, b_en_mpc = self._battery.bid_mpc(timestamp)
        a_s = torch.tensor([
            b_fcas_mpc.power_signed(),
            b_fcas_mpc.price(),
            b_en_mpc.power_signed()
        ])
        a_a = self._actor_nn(state)
        a_e = torch.randn(1, 3)
        return a_s, a_a, a_e, (1 - k) * a_s + k * (a_a + a_e)

    def _compute_reward(self, b_fcas, b_en, en_cleared_price,
                        b_fcas_cleared, r_shield):
        # assume the markets are pay-as-b
        # assume the en market always clears your b
        b_en_cleared_power = b_en.power_signed()
        b_en_cleared_price = b_en.price()

        b_fcas_price = b_fcas.price()
        b_fcas_power = b_fcas.power_signed()
        b_fcas_cleared_power = b_fcas_cleared.power()
        b_fcas_cleared_price = b_fcas_cleared.price()

        # bare bones r function
        r = -(
            b_en_cleared_power + abs(b_fcas_cleared_power)
        ) * b_en_cleared_price + b_fcas_cleared_power * b_fcas_cleared_price

        soe = self._battery.get_soe()
        total_capacity = self._battery._total_capacity
        max_power = self._battery._max_power
        max_ramp = self._battery._max_ramp

        new_en = soe + self._battery._efficiency * b_en_cleared_power +\
                self._battery._ratio_fcast * b_fcas_cleared_power

        # weight the constraints by how 'much' the constraint
        # is violated multiplied by some scalar. this can be changed.
        # only punish if bounds on capacity, power, or ramp are violated.
        penalty = 100

        if new_en > total_capacity:
            r -= penalty * (new_en - total_capacity)
        if new_en < 0:
            r -= penalty * (-new_en)
        if -b_fcas_power > max_ramp:
            r -= penalty * b_fcas_power

        # penalize "low" fcas bs
        if b_fcas_power > 0:
            r -= penalty * b_fcas_power
        if -b_fcas_power > max_ramp:
            r -= penalty * abs(-b_fcas_power - max_ramp)
        if -b_en_cleared_power > max_power:
            r -= penalty * abs(-b_en_cleared_power - max_power)

        if b_fcas_price < 0:
            r -= penalty * abs(b_fcas_price)

        return r

    def _get_step_of_day(self, timestamp, timestep_min=30):
        return timestamp.hour * 60 / timestep_min + \
                timestamp.minute / timestep_min


def save_data(b_fcas, b_en, b_fcas_cleared,
              fcas_clearing_price, soe, index, timestamp, en_price, r,
              current_state_value, next_state_value, raise_demand,
              b_fcas_actor_power, b_fcas_actor_price,
              b_en_actor_power, k, delta):
    d = {}
    d['b_fcas_power'] = b_fcas.power()
    d['b_fcas_price'] = b_fcas.price()
    d['b_fcas_type'] = b_fcas.type()

    d['b_fcas_actor_power'] = b_fcas_actor_power
    d['b_fcas_actor_price'] = b_fcas_actor_price
    d['b_en_actor_power'] = b_en_actor_power

    if b_en.type() == 'gen':
        d['b_en_power_gen'] = b_en.power_signed()
        d['b_en_power_load'] = 0
    else:
        d['b_en_power_gen'] = 0
        d['b_en_power_load'] = b_en.power_signed()

    d['b_en_price'] = b_en.price()
    d['b_en_type'] = b_en.type()
    d['b_fcas_cleared_power'] = b_fcas_cleared.power_signed()
    d['fcas_clearing_price'] = fcas_clearing_price
    d['en_price'] = en_price
    d['r'] = r
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
    f = open('hybrid_rl_results.csv', "w+")
    f.close()
    actor_critic = ActorCritic()
    actor_critic.run_simulation()


if __name__ == '__main__':
    main()
