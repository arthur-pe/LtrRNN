import torch
from torch import nn
import numpy as np

from ltrRNN.sde.base import DynamicalSystem


def additional_initial_states(func):
    """
    :param func: get_initial_state(self, batch_size) function
    :return: a decorator which adds additional possible initial states
    """

    def decorator(self, batch_size):
        temp = func(self, batch_size) #in case some parameterization happen inside get_initial_state
        if self.init == 'zero': return torch.zeros((batch_size, self.dim), device=self.device)
        elif self.init == 'randn': return torch.randn((batch_size, self.dim), device=self.device)/20
        elif self.init == 'rand': return torch.rand((batch_size, self.dim), device=self.device)*2-1
        elif self.init == 'optimized': return self.initial_state.expand(batch_size, -1)
        else: return temp

    return decorator


class RNN(DynamicalSystem):

    def __init__(self, dim, in_dims=(), bias=False, time_constant=1.0, activation=torch.tanh,
                 init='randn', noise_dim=None, noise=0.0, device='cpu'):
        super().__init__(noise_dim=noise_dim, noise=noise, device=device)

        self.initial_state = nn.Parameter(torch.randn(dim, device=device)/20)

        if bias: self.b = nn.Parameter(torch.randn(dim, device=device)/np.sqrt(dim))

        self.W = nn.Parameter(torch.randn((dim,dim), device=device)/np.sqrt(dim))

        self.W_in = nn.ParameterList([nn.Parameter(torch.randn(dim, in_dim, device=device)/np.sqrt(sum(in_dims)))
                                      for in_dim in in_dims])

        self.dim = dim
        self.in_dims = in_dims
        self.bias = bias
        self.time_constant = time_constant
        self.activation = activation
        self.init = init

    def f(self, x, *args):

        inputs = self.activation(x) @ self.W + sum(u @ B.T for i, (B, u) in enumerate(zip(self.W_in, args)))

        if self.bias:
            inputs = inputs + self.b

        dx_dt = (inputs-x)*self.time_constant**-1

        return dx_dt

    @additional_initial_states
    def get_initial_state(self, batch_size):
        """
        :param batch_size:
        :return: tensor of shape (batch_size, self.dim)
        """
        return None
