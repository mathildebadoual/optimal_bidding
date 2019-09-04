import numpy as np


class Battery():
    def __init__(self):
        self.total_capacity = 129  # MWh
        self.max_power = 100  # MW
        self.efficiency = 0.98  # percent
        self.min_power = - 100  # MW
        self._energy = 50

    def step(self, power):
        self._energy += self.efficiency * power

    def get_energy(self):
        return self._energy
