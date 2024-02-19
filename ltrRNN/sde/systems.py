import torch
from torch import nn
import numpy as np


class System(nn.Module):
    def __init__(self, sdes, graph):
        """
        :param sdes: List of SDEs with methods: f, g, get_initial_state. (e.g. an instance of DynamicalSystem)
        :param graph: Graph representing their relationship. Such that graph_ij is != 0 is sde[i] feeds forward to sde[j].
            graph_ij is a nn.Module represents applying the module before passing it.
        """
        super(System, self).__init__()

        self.sdes = nn.ModuleList(sdes)
        self.graph = np.array(graph, dtype=object)
        self.dims = [i.dim for i in sdes]
        self.dim = sum(self.dims)

        self.adjacency_list = [np.where(i!=0)[0] for i in self.graph]
        self.graph_map = nn.ModuleList([nn.ModuleList([nn.Identity() if (j==0 or j==1) else j
                                                       for j in i]) for i in graph])

    def cut_states(self, x):

        temp = []
        current_index = 0
        for i in self.dims:
            temp.append(x[..., current_index:current_index+i])
            current_index += i

        return temp

    def get_initial_state(self, batch_size):
        return torch.cat([i.get_initial_state(batch_size) for i in self.sdes], dim=-1)

    def f_all(self, x):

        cut_states = self.cut_states(x)
        derivatives = []

        for i in range(len(self.adjacency_list)):

            derivative = self.sdes[i].f(*[self.graph_map[i][j](cut_states[j]) for j in self.adjacency_list[i]])
            derivatives.append(derivative)

        return torch.cat(derivatives, dim=-1)

    def g_all(self, x):

        cut_states = self.cut_states(x)
        derivatives = []

        for i in range(len(self.adjacency_list)):

            derivative = self.sdes[i].g(*[self.graph_map[i][j](cut_states[j]) for j in self.adjacency_list[i]])
            derivatives.append(derivative)

        return torch.cat(derivatives, dim=1)

    def build_parameterization(self):
        self.p = []
        for s in self.sdes:
            parameter = s.get_parameterization()
            if parameter is not None:
                if isinstance(parameter[0], tuple):
                    self.p.append(*parameter)
                else: # Single parameter
                    self.p.append(parameter)

    def backward_parameterization(self):
        for p in self.p: p[1].backward(p[0].grad)


class SDE(System):
    """
    Wrapper for the system including the required functions for torchsde.
    """

    def __init__(self, sdes, graph, sde_type='stratonovich', noise_type='diagonal'):

        super(SDE, self).__init__(sdes, graph)

        self.sde_type = sde_type
        self.noise_type = noise_type

    def f(self, t, x):

        return self.f_all(x)

    def g(self, t, x):

        return self.g_all(x)
