import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import random


# parameters:
p_1 = 2.5
p_2 = 2.5
T_1 = 2
T_2 = 2
M_1 = 1
M_2 = 20
r_max = 4
r_min = 0
MAX_TIME = 200

y_init = np.array([0, 0, 0, 0])
time = np.linspace(0, MAX_TIME, MAX_TIME+1)
d = np.array([random.uniform(0, 0.5) for t in time])
y0_record = []
y1_record = []
u1_record = []
u2_record = []

timer = 0


class State:
    def __init__(self, u_1, u_2, t_1, t_2, transition_to, name):
        self.u_1 = u_1
        self.u_2 = u_2
        self.t_1 = t_1
        self.t_2 = t_2
        self.transition_to = transition_to
        self.state_to_transit = None
        self.name = name

    def check_transition_from(self):
        reachable_states = transition[self]
        for state in reachable_states:

            if state.check_transition_to(self.y_current):
                self.state_to_transit = state
                return True
        return False

    def check_transition_to(self, y):
        return self.transition_to(y)

    def run_step(self):
        y0 = self.y_current[0] + self.u_1 + self.u_2 - self.y_current[1]
        if self.y_current[1] >= r_max - 0.5 :
            y1 = self.y_current[1] - d[timer]
        elif self.y_current[1] <= r_min + 0.5 :
            y1 = self.y_current[1] + d[timer]
        else :
            y1 = self.y_current[1] + random.choice([-1, 1])*d[timer]
        y2 = self.y_current[2] + self.t_1
        y3 = self.y_current[3] + self.t_2
        self.y_current = np.array([y0, y1, y2, y3])
        return self.y_current

    def run_sim(self, y_init):
        self.y_current = y_init
        global timer
        global y0_record
        global y1_record
        global u1_record
        global u2_record
        while not self.check_transition_from() and timer <=  MAX_TIME:
            self.run_step()
            y0_record.append(self.y_current[0])
            y1_record.append(self.y_current[1])
            u1_record.append(M_2/p_1*self.u_1)
            u2_record.append(M_2/p_2*self.u_2)
            timer += 1
            print(self.name)
        self.y_current[2] = 0
        self.y_current[3] = 0
        if timer >= MAX_TIME:
            return "stop"
        return self.state_to_transit.run_sim(self.y_current)


def t_on_on(y):
    if y[2] == T_1 or y[3] == T_2:
        return True
    return False


def t_on_off(y):
    if y[0] >= M_2 - r_max:
        return True
    if y[2] == T_1:
        return True
    return False


def t_off_on(y):
    if y[0] >= M_2 - r_max:
        return True
    if y[3] == T_2:
        return True
    return False


def t_off_off(y):
    if y[0] >= M_2 - r_max:
        return True
    return False


def t_on_starts(y):
    if y[0] <= M_1 + (r_max - p_1)*T_2:
        return True
    return False


def t_off_starts(y):
    if y[0] <= M_1 + r_max*T_2 + (r_max - p_2)*T_1:
        return True
    return False


def t_starts_on(y):
    if y[0] <= M_1 + (r_max - p_2)*T_1:
        return True
    return False


def t_starts_off(y):
    if y[0] <= M_1 + r_max*T_1 + (r_max - p_1)*T_2:
        return True
    return False


on_on = State(p_1, p_2, 0, 0, t_on_on, 'on_on')
on_off = State(p_1, 0, 0, 0, t_on_off, 'on_off')
off_on = State(0, p_2, 0, 0, t_off_on, 'off_on')
off_off = State(0, 0, 0, 0, t_off_off, 'off_off')
on_starts = State(p_1, 0, 0, 1, t_on_starts, 'on_starts')
off_starts = State(0, 0, 0, 1, t_off_starts, 'off_starts')
starts_on = State(0, p_2, 1, 0, t_starts_on, 'starts_on')
starts_off = State(0, 0, 1, 0, t_starts_off, 'starts_off')

transition = {
        on_on: [on_off, off_on, off_off],
        on_off: [off_off, on_starts],
        off_on: [off_off, starts_on],
        off_off: [off_starts, starts_off],
        on_starts: [on_on],
        starts_on: [on_on],
        off_starts: [off_on],
        starts_off: [on_off]}

message = on_off.run_sim(y_init)
print(message)

plt.figure(figsize=(15, 8))
plt.title('Simulation of the steam boiler')
plt.xlabel('time')
plt.plot(time, y0_record, label='w')
plt.plot(time, y1_record, label='r')
plt.plot(time, u1_record, label='u1')
plt.plot(time, u2_record, label='u2')
plt.legend()
plt.grid()
plt.savefig('pbl_3.png')

