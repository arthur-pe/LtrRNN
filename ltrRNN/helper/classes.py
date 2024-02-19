import torch
from torch import nn
import numpy as np


class BatchWiseDNN(nn.Module):
    """
    Separate DNN for each batch dimension
    """

    def __init__(self, trials, in_dim, hidden_dim, out_dim, init=torch.randn, activation=torch.relu, device='cpu'):
        super(BatchWiseDNN, self).__init__()

        self.encoder_weight = nn.Parameter(init((trials, in_dim, hidden_dim[0]), device=device)/in_dim**0.5)
        self.encoder_bias = nn.Parameter(init((trials, 1, hidden_dim[0]), device=device) / in_dim ** 0.5)

        self.hidden_weight = nn.ParameterList([])
        self.hidden_bias = nn.ParameterList([])
        current_dim = hidden_dim[0]
        for i in range(1, len(hidden_dim)):
            self.hidden_weight.append(nn.Parameter(init((trials, current_dim, hidden_dim[i]), device=device)/current_dim**0.5))
            self.hidden_bias.append(nn.Parameter(init((trials, 1, hidden_dim[i]), device=device)/current_dim**0.5))

            current_dim = hidden_dim[i]

        self.decoder_weight = nn.Parameter(init((trials, current_dim, out_dim), device=device) / current_dim ** 0.5)
        self.decoder_bias = nn.Parameter(init((trials, 1, out_dim), device=device) / current_dim ** 0.5)

        self.activation = activation
        self.hidden_dim = hidden_dim

    def forward(self, x):

        x = x.unsqueeze(1)
        x = self.activation(torch.bmm(x, self.encoder_weight)+self.encoder_bias)

        for i in range(len(self.hidden_dim)-1):
            x = self.activation(torch.bmm(x, self.hidden_weight[i])+self.hidden_bias[i])

        x = torch.bmm(x, self.decoder_weight)+self.decoder_bias

        return x.squeeze(1)


class ConditionWiseControlMap(nn.Module):
    """
    Map from the condition space to the trial space.
    """

    def __init__(self, condition, hidden_dim, output_dim, activation=torch.tanh, device='cpu'):

        super(ConditionWiseControlMap, self).__init__()

        self.condition = condition
        self.activation = activation

        self.number_conditions = torch.max(self.condition)+1
        self.device = device

        self.to_low_dim = nn.Parameter(torch.randn(output_dim, hidden_dim, device=device)/np.sqrt(hidden_dim))

        self.build_trial_condition_matrix()

    def build_trial_condition_matrix(self):

        self.trials_to_condition = torch.zeros(len(self.condition), self.number_conditions, device=self.device)
        self.trials_to_condition[np.arange(0, len(self.condition), 1), self.condition] = 1

    def h(self, control):

        temp = (control.transpose(-2,-1)[...,:self.number_conditions] @ self.trials_to_condition.T).transpose(-2,-1)
        temp2 = self.to_low_dim / torch.sqrt(torch.sum(self.to_low_dim**2, dim=1)).unsqueeze(dim=1)

        return temp @ temp2.T

    def forward(self, control):
        return self.activation(self.h(control))


class LinearTanh(nn.Module):
    """
    One layer perceptron with tanh activition.
    """

    def __init__(self, in_dim, out_dim, bias=False, device='cpu'):
        super().__init__()

        self.W = nn.Parameter((2*torch.rand(out_dim, in_dim, device=device)-1)/np.sqrt(in_dim))

        if bias:
            self.b = nn.Parameter((2*torch.rand(out_dim, device=device)-1)/np.sqrt(in_dim))

        self.bias = bias
        self.device = device

    def forward(self, x):

        temp = torch.tanh(x @ self.W.T)
        if self.bias: temp = temp + self.b

        return temp


class TrialRankMap(nn.Module):
    """
    Takes a time x r1 x n1 control and returns time x trial x n2
    """

    def __init__(self, neuron_in_dim, neuron_out_dim, trial_rank, trial_out, bias=False, activation=torch.tanh, device='cpu'):
        super().__init__()

        self.neuron_map = nn.Linear(neuron_in_dim, neuron_out_dim, bias=bias, device=device)
        self.trial_map = nn.Linear(trial_rank, trial_out, bias=False, device=device)

        self.rank = trial_rank
        self.activation = activation

    def forward(self, x):

        x = self.neuron_map(x)
        x = self.trial_map(x[..., :self.rank, :].transpose(-2, -1)).transpose(-2, -1)

        return self.activation(x)


class OrthogonalScaled(nn.Module):
    """
    Orthogonal linear decoder.
    """

    def __init__(self, in_dim, out_dim, bias=True, scaling=True, scaling_neuron_wise=False, device='cpu'):
        super(OrthogonalScaled, self).__init__()

        self.M = nn.utils.parametrizations.orthogonal(nn.Linear(in_dim, out_dim, device=device, bias=False),
                                                      orthogonal_map='matrix_exp')

        if bias:
            self.b = nn.Parameter((torch.rand(out_dim, device=device)*2-1)/np.sqrt(in_dim))
        if scaling:
            self.c = nn.Parameter(torch.tensor(1.0, device=device))
        if scaling_neuron_wise:
            self.c = nn.Parameter(torch.ones(out_dim, device=device))
            scaling = True

        self.bias = bias
        self.scaling = scaling
        self.scaling_neuron_wise = scaling_neuron_wise

        self.weight = self.M.weight

    def forward(self, x):

        self.weight = self.M.weight

        temp = self.M(x)

        if self.scaling: temp = temp * self.c
        if self.bias: temp = temp + self.b

        return temp
