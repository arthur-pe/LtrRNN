import torch
from torch import nn
import warnings


class DynamicalSystem(nn.Module):

    def __init__(self, sde_type='stratonovich', noise_type='diagonal', noise_dim=None, noise=0.0, device='cpu'):
        super().__init__()

        self.sde_type = sde_type
        self.noise_type = noise_type
        self.noise_dim = noise_dim
        self.noise = noise
        self.device = device

    def f(self, x, *args):

        raise NotImplementedError(f"Module [{type(self).__name__}] is missing the required \"f\" method")

    def g(self, x, *args):
        """
        The g function of the sde dx = f(x)dt+g(x)dW

        :param x: tensor of shape (batch_size, self.dim)
        :return: tensor of shape (batch_size, self.dim)
        """

        return torch.full(list(x.shape), self.noise, device=self.device)

    def g_general(self, x, *args):
        """
        The g function of the sde dx = f(x)dt+g(x)dW

        :param x: tensor of shape (batch_size, self.dim)
        :return: tensor of shape (batch_size, self.dim, self.noise_dim)
        """

        return torch.full(list(x.shape)+[self.noise_dim], self.noise, device=self.device)

    def get_initial_state(self, batch_size):
        """
        Must be implemented in classes inheriting from DynamicalSystem.

        :param batch_size:
        :return: tensor of shape (batch_size, self.dim), the initial state of the dynamical system upon calling sdeint.
        """

        raise NotImplementedError(f"Module [{type(self).__name__}] is missing the required \"get_initial_state\" method")

    def get_parameterization(self):
        """
        Returns a list (or one element) of parameters of the dynamical system which require a parameterization.
        Each element of the list contains two copies of the parameter:
        -one which is used for evaluating the sde, whose gradient w.r.t. the parameterization is 0.
        -one which keeps the gradient of the parameterization.
        After the sde is evaluated, the gradient of the first is pushed backward to the second.
        By default, a class inherited from DynamicalSystem needs not have a parameterization, in which case None
        is returned.

        :return: None
        """

        return None
