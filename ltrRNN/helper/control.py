from .classes import *
from ltrRNN.sde.base import *
from ltrRNN.rnn_architectures.base_rnn import additional_initial_states

import torch
from torch import nn
import numpy as np


class Controller(DynamicalSystem):
    """
    Separate controller for each batch (aka trial) dimension.
    """

    def __init__(self, dim, trials, dnn_hidden_dims, in_dims=(), time_constant=1.0, activation=torch.tanh,
                 init='randn', noise_dim=None, noise=0, device='cpu'):
        super(Controller, self).__init__(noise_dim=noise_dim, noise=noise, device=device)

        self.dnn = BatchWiseDNN(trials, dim, dnn_hidden_dims, dim, device=device)
        self.W_in = nn.ParameterList([nn.Parameter(torch.randn(dim,in_dim,
                                                               device=device)/np.sqrt(sum(in_dims))) for in_dim in in_dims])
        self.initial_state = nn.Parameter(torch.randn(trials, dim, device=device)/20)

        self.dim = dim
        self.trials = trials
        self.in_dims = in_dims
        self.time_constant = time_constant
        self.activation = activation
        self.init = init

    @additional_initial_states
    def get_initial_state(self, batch_size):
        return self.initial_state

    def f(self, x, *args):

        inputs = sum(u @ self.W_in[i].transpose(0,1) for i, u in enumerate(args))
        return (self.activation(self.dnn(x)+inputs)-x)*self.time_constant**-1


class TrialRankController(DynamicalSystem):
    """
    Separate controller for each low d. trial_dimension (e.g. condition dimension), which projects back the full_trial_dimension.
    """

    def __init__(self, dim, full_trial_dimension, trial_dimension, dnn_hidden_dims, in_dims=(), time_constant=1.0,
                 activation=torch.tanh, init='randn', noise_dim=None, noise=0, device='cpu'):
        super(TrialRankController, self).__init__(noise_dim=noise_dim, noise=noise, device=device)

        self.dnn = BatchWiseDNN(trial_dimension, dim, dnn_hidden_dims, dim, device=device)
        self.W_in = nn.ParameterList([nn.Parameter(torch.randn(dim, in_dim,
                                                               device=device) / np.sqrt(in_dim)) for in_dim in in_dims])

        self.initial_state = nn.Parameter((torch.rand(trial_dimension, dim, device=device)*2-1))

        self.full_trial_dimension = full_trial_dimension
        self.trial_dimension = trial_dimension
        self.dim = dim

        self.time_constant = time_constant
        self.activation = activation
        self.init = init

    def get_initial_state(self, batch_size):
        match self.init:
            case 'optimized':
                return torch.cat([self.initial_state, torch.zeros([self.full_trial_dimension-self.trial_dimension,
                                                           self.dim], device=self.device)], dim=0)
            case 'zero':
                return torch.zeros([self.full_trial_dimension, self.dim], device=self.device)
            case 'randn':
                return torch.randn([self.full_trial_dimension, self.dim], device=self.device)/10
            case 'rand':
                return torch.rand([self.full_trial_dimension, self.dim], device=self.device)*2-1

    def f(self, x, *args):

        x, trash = x[:self.trial_dimension], x[self.trial_dimension:]

        inputs = sum(u @ self.W_in[i].transpose(0,1) for i, u in enumerate(args))
        temp = (self.activation(self.dnn(x)*5+inputs)-x)*self.time_constant**-1

        return torch.cat([temp,torch.zeros_like(trash)])
