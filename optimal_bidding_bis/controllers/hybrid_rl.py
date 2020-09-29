import sys
import os
import pandas as pd
import torch
import torch.optim as optim
import numpy as np

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))))

from utils.data_postprocess import DataProcessor
from utils.nets import ActorNet, CriticNet, Normalizer
from environments.agents import Battery, Bid
from environments.energy_market import EnergyMarket

RESULTS_PATH = 'results/hybrid_results.csv'

last_day_of_data = pd.Timestamp(
    year=2018,
    month=11,
    day=1,
    hour=0,
    minute=0,
)
filename = 'FiveMonths2018_30min.csv'
data_utils = DataProcessor(last_day_of_data, filename)


class ActorCritic():
    def __init__(self, start_time, end_time):
        # hyperparameters
        self._exploration_size = None

        self._actor_step_size = 10e-4
        self._discount_factor = 0.98

        torch.manual_seed(1)
        self._energy_market = EnergyMarket(
            start_time,
            end_time
            )
        self._battery = Battery()
        self._actor_nn = ActorNet()
        self._critic_nn = CriticNet()
        self._criterion = torch.nn.MSELoss()
        self._normalizer = Normalizer(4)

        self._optimizer_actor = optim.Adagrad(self._actor_nn.parameters(),
                                              lr=5* 10e-4)
        self._optimizer_critic = optim.Adagrad(self._critic_nn.parameters(),
                                           lr=10e-4)

    def run_simulation(self):
        end = False
        index = 0
        k = 0
        # Initialize to 0 (only affects first state)
        clearing_price = None
        next_state = None

        while not end:
            timestamp = self._energy_market.get_timestamp()
            print('---------- timestamp: %s ----------' % timestamp)

            # en_cleared_price will hold previous clearing price
            # unless it is the first timestamp
            if clearing_price is None:
                clearing_price = data_utils.get_energy_price(timestamp)
                # create the state:
                soe = self._battery.get_soe()
                step_of_day = self._get_step_of_day(timestamp)
                demand = data_utils.get_energy_demand(timestamp)
                state = torch.tensor([
                    step_of_day, soe, clearing_price,
                    demand
                    ])
            else:
                state = next_state

            # compute the action = [p_raise, c_raise, p_en]
            a_s, a_a, a_e, action_composite = self._compute_action(
                state, timestamp, k, index)

            # update the actor
            loss_actor_sup = self._update_actor_supervised(a_s.float(),
                                                           a_a.float())

            bid = self._transform_to_bid(action_composite)

            # bid, r_shield = self._shield(bid, a_s)

            # run the market dispatch
            bid_cleared, clearing_price, end, demand = self._energy_market.step(
                bid)

            r = self._compute_reward(bid, clearing_price,
                                     bid_cleared)
            # self._normalizer.observe(r)
            # r = self._normalizer.normalize(r)

            # update soe of the battery with the cleared power
            self._battery.step(bid_cleared.power_signed())

            print('reward: %s' % r)
            next_soe = self._battery.get_soe()

            next_timestamp = timestamp + pd.Timedelta('30 min')
            next_step_of_day = self._get_step_of_day(next_timestamp)
            next_demand = data_utils.get_energy_demand(
                next_timestamp)
            #next_en_cleared_price = data_utils.get_energy_price(timestamp)
            next_state = torch.tensor([
                next_step_of_day, next_soe,
                clearing_price,
                next_demand
            ])

            # update eligibility
            self._normalizer.observe(state)
            state = self._normalizer.normalize(state)
            self._normalizer.observe(next_state)
            next_state = self._normalizer.normalize(next_state)

            value = self._critic_nn(state.float())
            next_value = self._critic_nn(next_state.float())
            eligibility = r + self._discount_factor * next_value - value
            print('eligibility: %s' % eligibility)

            # update neural nets
            loss_critic = self._update_critic(eligibility.float())
            print('state: %s' % state)
            value = self._critic_nn(state.float())
            print('value: %s' % value)
            loss_actor = self._update_actor(value.float())

            # save data
            a_a_numpy = a_a.data.numpy()
            bid_actor_power = abs(a_a_numpy[0])
            bid_actor_price = a_a_numpy[1]
            save_data(bid, bid_cleared,
                      clearing_price, next_soe, index, timestamp,
                      r, value, demand,
                      next_value,
                      bid_actor_power, bid_actor_price,
                      k, eligibility,
                      loss_actor_sup, loss_actor, loss_critic)

            index += 1
            if index > 400:
                if index % 10 == 0 and k < 0.5:
                    k += 0.005
                    print(k)

    def _update_critic(self, eligibility):
        loss = self._criterion(eligibility, torch.zeros(len(eligibility)))
        print('loss critic: %s' % loss)
        # self._optimizer_critic.zero_grad()
        loss.backward()
        self._optimizer_critic.step()
        return loss

    def _update_actor_supervised(self, a_s, a_a):
        loss = self._criterion(a_a, a_s)
        print('loss actor suppervised: %s' % loss)
        # self._optimizer_actor.zero_grad()
        loss.backward()
        self._optimizer_actor.step()
        return loss

    def _update_actor(self, value):
        loss = self._criterion(value, torch.zeros(len(value)))
        # self._optimizer_actor.zero_grad()
        loss.backward()
        # self._optimizer_actor.step()
        for f in self._actor_nn.parameters():
            f.data.sub(- f.grad.data * self._actor_step_size)
        return loss

    def _transform_to_bid(self, action):
        action = action.data.numpy()
        print(action)
        if action[0] >= 0:
            type = 'load'
        else:
            type = 'gen'
        bid = Bid(abs(action[0]), action[1], bid_type=type)
        return bid

    # def _shield(self, b_fcas, b_en, a_s):
    #     r = 0
    #     if b_fcas.power() < 0:
    #         b_fcas = Bid(-a_s[0], a_s[1], bid_type='gen')
    #         r -= 100 * abs(b_fcas.power())
    #     return b_fcas, b_en, r

    def _compute_action(self, state, timestamp, k, index):
        # timestamp = datetime.fromtimestamp(timestamp)
        bid_mpc = self._battery.bid_mpc(timestamp)
        a_s = torch.tensor([
            bid_mpc.power_signed(),
            bid_mpc.price()
        ])
        a_a = self._actor_nn(state.float())
        a_e = torch.tensor([
            0, 0
        ])
        q = 1
        a_e = torch.tensor([
            q * np.random.uniform(low=-1, high=1),
            q * np.random.uniform(low=-1, high=1),
        ], dtype=torch.float64)
        return a_s, a_a, a_e, (1 - k) * a_s + k * (a_a + a_e)

    def _compute_reward(self, bid, clearing_price,
                        bid_cleared):
        # assume the markets are pay-as-b
        # assume the en market always clears your b
        bid_cleared_power = bid_cleared.power_signed()
        bid_cleared_price = bid_cleared.price()
        bid_power = bid.power_signed()
        bid_price = bid.price()

        print('bid type = %s' % bid_cleared.type())
        print('bid power = %s' % bid_cleared_power)
        print('bid price = %s' % bid_cleared_price)

        # bare bones r function
        if bid_cleared.type() == 'gen':
            r = (- bid_cleared_power) * bid_price / 100
        else:
            r = (- bid_cleared_power) * clearing_price / 100
        print('bid*price= %s' % r)

        soe = self._battery.get_soe()
        total_capacity = self._battery._total_capacity
        max_power = self._battery._max_power

        new_en = soe + self._battery._efficiency * bid_power

        # weight the constraints by how 'much' the constraint
        # is violated multiplied by some scalar. this can be changed.
        # only punish if bounds on capacity, power, or ramp are violated.
        penalty = 1
        if new_en > total_capacity:
            print('> capacity max')
            print('soe: %s' % soe)
            r -= penalty * (new_en - total_capacity)
        if new_en < 0:
            print('< capacity min')
            print('soe: %s' % soe)
            r -= penalty * (- new_en)
        if bid_price < 0:
            print('< bid price min')
            r = - penalty * abs(bid_price)

        # penalize "low" bids
        if abs(bid_power.round(2)) > max_power:
            print('> bid power capacity')
            r -= penalty * abs(- bid_power - max_power)

        return r

    def _get_step_of_day(self, timestamp, timestep_min=30):
        return timestamp.hour * 60 / timestep_min + \
            timestamp.minute / timestep_min


def save_data(bid, bid_cleared,
              clearing_price, soe, index, timestamp,
              r, value, demand, next_value,
              bid_actor_power, bid_actor_price,
              k, eligibility,
              loss_actor_sup, loss_actor, loss_critic):
    d = {}

    if type(bid.price()) == torch.Tensor:
        d['battery_bid_price'] = bid.price().item()
    else:
        d['battery_bid_price'] = bid.price()
    d['bid_type'] = bid.type()

    if bid.type() == 'gen':
        d['battery_bid_power_gen'] = bid.power_signed()
        d['battery_bid_power_load'] = 0
    else:
        d['battery_bid_power_gen'] = 0
        d['battery_bid_power_load'] = bid.power_signed()

    if (type(bid.price()) == torch.Tensor or type(bid.power()) == torch.Tensor):
        d['r'] = r.item()
    else:
        d['r'] = r

    if bid_cleared.type() == 'gen':
        d['battery_bid_cleared_power_gen'] = bid_cleared.power_signed()
        d['battery_bid_cleared_power_load'] = 0
    else:
        d['battery_bid_cleared_power_gen'] = 0
        d['battery_bid_cleared_power_load'] = bid_cleared.power_signed()
    d['current_state_value'] = value.item()
    d['next_state_value'] = next_value.item()
    d['bid_actor_power'] = bid_actor_power
    d['bid_actor_price'] = bid_actor_price

    d['demand'] = demand
    d['clearing_price'] = clearing_price
    d['soe'] = soe
    d['timestamp'] = timestamp
    d['k'] = k
    d['loss_actor'] = loss_actor.item()
    d['loss_critic'] = loss_critic.item()
    d['loss_actor_sup'] = loss_actor_sup.item()
    # d['delta'] = delta.data.numpy()

    df = pd.DataFrame(data=d, index=[index])
    with open(RESULTS_PATH, 'a') as f:
        if index == 0:
            df.to_csv(f, header=True)
        else:
            df.to_csv(f, header=False)


def main():
    # erase output csv
    f = open(RESULTS_PATH, "w+")
    f.close()

    # initialize data processing
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

    # data_utils = DataProcessor(last_day_of_data, 'FiveMonths2018_30min.csv')
    actor_critic = ActorCritic(start_time, end_time)
    actor_critic.run_simulation()


if __name__ == '__main__':
    if os.path.exists(RESULTS_PATH):
        os.remove(RESULTS_PATH)
    main()
