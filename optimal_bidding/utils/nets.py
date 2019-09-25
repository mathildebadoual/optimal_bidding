import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ActorNet(nn.Module):

    # timestep of day
    # soe
    # last raise clearing price
    # prediction of demand
    # clearing energy price
    # last cleared energy price maybeee
    # last week same day clearing raise price
    # yesterday same timestep clearing raise price
    # bids from other people - maybe later
    # 2 artificial time-dependent features.

    def __init__(self):
        super(ActorNet, self).__init__()

        self.fc1 = nn.Linear(6, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 8)
        self.fc4 = nn.Linear(8, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class CriticNet(nn.Module):

    # timestep of day
    # soe
    # last raise clearing price
    # prediction of demand
    # clearing energy price
    # last cleared energy price maybeee
    # last week same day clearing raise price
    # yesterday same timestep clearing raise price
    # bids from other people - maybe later
    # 2 artificial time-dependent features.

    def __init__(self):
        super(CriticNet, self).__init__()

        self.fc1 = nn.Linear(6, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 8)
        self.fc4 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class Layer:
    def __init__(self, weights_matrix, bias_vector, sigmoid_activation=True):
        self.weights_matrix = weights_matrix
        self.bias_vector = bias_vector
        self.sigmoid_activation = sigmoid_activation

    def compute_value(self, x_vector):
        result = np.add(np.dot(self.weights_matrix, x_vector),
                        self.bias_vector)
        if self.sigmoid_activation:
            result = np.exp(-result)
            result = 1 / (1 + result)
        return result

    def compute_value_without_activation(self, x_vector):
        result = np.add(np.dot(self.weights_matrix, x_vector),
                        self.bias_vector.reshape((-1, 1)))
        return result

    def compute_value_and_derivative(self, x_vector):
        if not self.sigmoid_activation:
            return (self.compute_value(x_vector), self.weights_matrix)
        temp = np.add(np.dot(self.weights_matrix, x_vector), self.bias_vector)
        temp = np.exp(-temp)
        value = 1.0 / (1 + temp)
        temp = temp / (1 + temp)**2
        #pre-multiplying by a diagonal matrix multiplies each row by
        #the corresponding diagonal element
        #(1st row with 1st value, 2nd row with 2nd value, etc...)
        jacobian = np.dot(np.diag(temp), self.weights_matrix)
        return (value, jacobian)


class Network:
    def __init__(self, layers):
        self.layers = layers

    def compute_value(self, x_vector):
        for l in self.layers:
            x_vector = l.compute_value(x_vector)
        return x_vector

    def compute_value_and_derivative(self, x_vector):
        x_vector, jacobian = self.layers[0].compute_value_and_derivative(
            x_vector)
        for l in self.layers[1:]:
            x_vector, j = l.compute_value_and_derivative(x_vector)
            jacobian = np.dot(j, jacobian)
        return x_vector, jacobian

    def compute_value_and_derivative_w(self, x_vector):
        list_vector_grad = []
        list_vector = []
        x_vector_ = x_vector.copy()
        list_vector.append(x_vector_.reshape((-1, 1)))
        for l in self.layers:
            x_vector_ = l.compute_value(x_vector_)
            list_vector.append(x_vector_.reshape((-1, 1)))
        print([a.shape for a in list_vector])
        n = len(list_vector)
        prev_l = None
        prev_grad = None
        for i, l in enumerate(reversed(self.layers)):
            print(i)
            d_sig = d_sigmoid(
                l.compute_value_without_activation(list_vector[n - i - 2]))
            list_grad_layer = []
            if i == 0:
                prev_grad = d_sig
                print(prev_grad.shape)
                for j in range(l.weights_matrix.shape[0]):
                    list_grad_layer.append(
                        np.dot(prev_grad[i], list_vector[n - i - 2].T))
                list_vector_grad.append(list_grad_layer)
            else:
                print(prev_grad.dot(prev_l.weights_matrix).shape)
                print(d_sig.shape)
                prev_grad = prev_grad.dot(prev_l.weights_matrix) * d_sig
                print(prev_grad.shape)
                for j in range(l.weights_matrix.shape[0]):
                    list_grad_layer.append(
                        np.dot(prev_grad[j].reshape((-1, 1)), list_vector[n - i - 2].T))
                list_vector_grad.append(list_grad_layer)
            prev_l = l
        return [g for g in reversed(list_vector_grad)]


def d_sigmoid(x):
    x = np.exp(-x)
    return x / (1 + x)**2
