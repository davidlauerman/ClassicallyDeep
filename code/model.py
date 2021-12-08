import math
import numpy as np
import os
import torch
import torch.nn

class DistributionLearner(torch.nn.Module):
    def __init__(self):
        super(DistributionLearner, self).__init__()

        self.window_size = 50
        self.input_size = 128
        self.batch_size = 12
        self.hidden_size = self.input_size
        self.num_layers = 3
        self.learning_rate = 0.001

        self.GRU = torch.nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first = True)

        self.linear = torch.nn.Sequential(torch.nn.Linear(self.hidden_size, self.hidden_size),
                                                 torch.nn.ReLU(),
                                                 torch.nn.Linear(self.hidden_size, self.hidden_size),
                                                 torch.nn.ReLU(),
                                                 torch.nn.Linear(self.hidden_size, self.hidden_size),
                                                 torch.nn.ReLU()) # change to torch.nn.Softmax() potentially

    def call(self, inputs, h0):
        """
        Performs a forward pass for the RNN
        input: batch of input data
        """
        out, hidden_state = self.GRU(inputs, h0)
        return self.linear(out), hidden_state





class QuantityLearner(torch.nn.Module):
    def __init__(self):
        super(QuantityLearner, self).__init__()

        self.window_size = 50
        self.input_size = 1
        self.batch_size = 12
        self.hidden_size = 1
        self.num_layers = 3
        self.learning_rate = 0.001
        self.concurrent_notes = 20

        self.GRU = torch.nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first= True)
        self.linear = torch.nn.Linear(self.hidden_size, self.hidden_size)

    def call(self, inputs, h0):
        """
        Performs a forward pass for the RNN
        input: batch of input data
        """
        out, hidden_state = self.GRU(inputs, h0)
        return self.linear(out), hidden_state
