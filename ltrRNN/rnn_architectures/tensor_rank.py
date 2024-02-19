from ltrRNN.rnn_architectures.base_rnn import *

import torch


def trial_wise_matmul(A, B):
    return torch.einsum('...ji,...j->...i', [A, B])


class TensorRank(DynamicalSystem):
    """
    Base class for low tensor rank RNNs
    """

    def __init__(self,
                 dim,
                 rank,
                 trials,
                 optimize_input_maps=True,
                 in_dims=(),
                 in_space_inputs=False,
                 condition=None,
                 bias=False,
                 noise=0.0,
                 noise_dim=None,
                 time_constant=1.0,
                 activation=torch.tanh,
                 device='cpu'):
        """
        :param dim: (int) neuron dimension of the RNN
        :param rank: (int) rank
        :param trials: (int) trial dimension
        :param optimize_input_maps: (bool) Whether to optimize the maps from the input to neuron space
        :param in_dims: (list of int) dimensions of the inputs
        :param in_space_inputs: (bool) whether the inputs are constrained to the space spanned by the columns
        :param condition: (torch.Tensor, trials)  If provided, the initial state is condition-specific
        :param bias: (bool) Whether to have a bias term (constrained to the column space)
        :param noise: (float) white noise strength
        :param noise_dim: (int) necessary for DynamicalSystem
        :param time_constant: Time constant of the RNN
        :param activation: activation function
        :param device: torch device
        """


        super().__init__(noise_dim=noise_dim, noise=noise, device=device)

        self.condition = torch.zeros(trials, device=device, dtype=torch.long) if condition is None else condition
        self.unique_condition = torch.unique(self.condition)
        self.condition_trial_vector = torch.stack([(self.condition == x) for x, i in enumerate(self.unique_condition)], dim=-1).float()

        coef_init = 0.2
        self.initial_state_linear_combination = nn.Parameter(np.sqrt(dim)*coef_init*(torch.rand((len(self.unique_condition), rank), device=device)*2-1) / np.sqrt(rank))

        self.W_in_coef = nn.ParameterList([nn.Parameter(torch.ones(in_dim, device=device)*np.sqrt(dim/sum(in_dims))) for in_dim in in_dims])

        self.bias_coef = nn.Parameter((torch.rand(rank, device=device)*2-1) / np.sqrt(rank))

        if in_space_inputs:
            self.column_combination_input_map = nn.ParameterList(
                [nn.Parameter((torch.rand(in_dim, rank, device=device)*2-1)/np.sqrt(dim/sum(in_dims)/rank)) for in_dim in in_dims])

        self.dim = dim
        self.rank = rank
        self.trials = trials
        self.optimize_input_maps = optimize_input_maps
        self.in_dims = in_dims
        self.in_space_inputs = in_space_inputs

        self.time_constant = time_constant
        self.activation = activation
        self.bias = bias

        self.cum_dims = np.cumsum([0]+list(self.in_dims))

        self.init_weight = nn.Parameter(torch.tensor(1.0, device=device), requires_grad=False)
        self.init_bias = nn.Parameter(torch.tensor(0.0, device=device), requires_grad=False)

        self.define_parameters()

    def define_parameters(self):

        self.W_column = nn.Linear(self.rank + (sum(self.in_dims) if not self.in_space_inputs else 0), self.dim, bias=False, device=self.device)
        self.W_row = nn.Linear(self.rank, self.dim, bias=False, device=self.device)
        self.W_trial = nn.Linear(self.rank, self.trials, bias=False, device=self.device)
        with torch.no_grad():
            self.W_column.weight.copy_(torch.randn_like(self.W_column.weight)/np.sqrt(self.dim))
            self.W_trial.weight.copy_(0.2*(-1 + 2 * torch.rand(self.trials, self.rank, device=self.device))/np.sqrt(self.rank))
            self.W_row.weight.copy_(self.W_column.weight[:,:self.rank])

    def construct_weight(self):

        W = torch.einsum('ir,jr,kr->ijk', [self.W_trial.weight+1,
                                           self.W_row.weight/torch.norm(self.W_row.weight, dim=0).unsqueeze(0),
                                           self.W_column.weight[:,:self.rank]/torch.norm(self.W_column.weight[:,:self.rank], dim=0).unsqueeze(0)])

        return W

    def construct_input_weight(self):

        if not self.in_space_inputs:
            W_in = [self.W_column.weight[:,self.rank+self.cum_dims[i]:self.rank+self.cum_dims[i+1]] *
                         self.W_in_coef[i].unsqueeze(0) for i in range(len(self.cum_dims)-1)]
        else:
            W_in = [self.W_column.weight[:,:self.rank] * self.W_in_coef[i] for i in range(len(self.cum_dims)-1)]

        if not self.optimize_input_maps:
            for wi in range(len(W_in)):
                W_in[wi] = W_in[wi].detach()

        return W_in

    def get_components(self, *args, **kwargs):

        return [self.W_trial.weight.T+1, self.W_row.weight.T, self.W_column.weight.T]

    def construct_bias(self):

        return (self.bias_coef.unsqueeze(0) @ self.W_column.weight[:,:self.rank].T).squeeze(0)

    def get_parameterization(self):

        W = self.construct_weight()

        W_in = self.construct_input_weight()

        b = self.construct_bias()

        self.W = nn.Parameter(W.detach())
        self.W_in = nn.ParameterList([i.detach() for i in W_in])
        self.b = nn.Parameter(b.detach())

        return [self.W, W] + [self.b, b] + [[i,j] for i,j in zip(self.W_in, W_in)]

    def get_initial_state(self, batch_size):

        initial_state = (self.initial_state_linear_combination @ self.W_column.weight[:, :self.rank].T)
        initial_state = self.condition_trial_vector @ initial_state

        return initial_state

    def f(self, x, *args):

        inputs = trial_wise_matmul(self.W, self.activation(x))
        inputs = inputs + sum(u @ self.W_in[i].T for i, u in enumerate(list(args)))
        if self.bias: inputs = inputs + self.b

        return (inputs-x)*self.time_constant**-1

class RationalQuadratic:
    def __init__(self, l, sigma, alpha=1.0):

        self.l = l
        self.sigma = sigma
        self.alpha=alpha

    def __call__(self, dx):
        return (self.sigma**2)*(1+dx ** 2 / (2 * self.alpha * self.l ** 2))**(-self.alpha)


class TensorRankSmooth(TensorRank):
    """
    Low tensor rank RNN with smoothness constraint over trials.
    """
    def __init__(self, dim, rank, trial_ids, test_mask, kernel, epoch=None, std_observation=1,
                 optimize_input_maps=True, in_dims=(), in_space_inputs=False, condition=None, bias=False, noise=0.0, noise_dim=None, time_constant=1.0,
                 activation=torch.tanh, device='cpu'):
        """
        :param trial_ids: (torch.Tensor, trials) the trial ids to use as kernel independent variables
        :param test_mask: (torch.Tensor, trials) whether to retain some trials for testing
        :param kernel: (nn.Module) trial covariance kernel
        :param epoch: (torch.Tensor, trials) If not None, makes the kernel discontinuous at changes of epochs.
        :param std_observation: (float) std of observation noise
        """

        self.kernel = kernel

        self.covariance_matrix = self.kernel(trial_ids.unsqueeze(-1) - trial_ids.unsqueeze(0))

        if epoch is None: epoch = torch.zeros_like(trial_ids)

        block_diag = torch.zeros_like(self.covariance_matrix, dtype=torch.bool)
        for e in torch.unique(epoch):
            temp = (epoch == e)
            temp = torch.outer(temp, temp)
            block_diag = block_diag | temp

        self.covariance_matrix = self.covariance_matrix * block_diag

        self.std_observation = std_observation
        self.test_mask = test_mask if test_mask is not None else torch.zeros(len(trial_ids), dtype=torch.bool, device=device)

        super().__init__(dim, rank, len(trial_ids), optimize_input_maps, in_dims, in_space_inputs, condition, bias, noise, noise_dim,
                         time_constant, activation, device)

        self.observation_bias = nn.Parameter(torch.ones(rank, device=device))

    def define_parameters(self):
        self.W_column = nn.Linear(self.rank + sum(self.in_dims), self.dim, bias=False, device=self.device)
        self.W_row = nn.Linear(self.rank, self.dim, bias=False, device=self.device)
        self.W_trial = nn.Linear(self.rank, self.trials, bias=False, device=self.device)

        with torch.no_grad():
            self.W_column.weight.copy_(torch.randn_like(self.W_column.weight)/np.sqrt(self.dim))
            self.W_trial.weight.copy_(torch.randn(self.trials, self.rank, device=self.device))
            self.W_row.weight.copy_(self.W_column.weight[:,:self.rank])

    def construct_trial_weight(self):

        temp = self.W_trial.weight

        cov = torch.linalg.cholesky(self.covariance_matrix+10**-6*torch.eye(self.trials, device=self.device))

        W_trial_smooth = cov @ (temp * ~self.test_mask.unsqueeze(1))

        return W_trial_smooth + self.observation_bias.unsqueeze(0)

    def construct_trial_weight_observation(self):

        temp = self.W_trial.weight
        cov = torch.linalg.cholesky(self.covariance_matrix +
                                    torch.eye(self.trials, device=self.device)*self.std_observation**2)

        W_trial_smooth = cov @ (temp * ~self.test_mask.unsqueeze(1))

        return W_trial_smooth + self.observation_bias.unsqueeze(0)

    def construct_weight(self, observation=True):

        W_trial_smooth = self.construct_trial_weight_observation() if observation else self.construct_trial_weight()

        W = torch.einsum('ir,jr,kr->ijk', [W_trial_smooth,
                                           self.W_row.weight / torch.norm(self.W_row.weight, dim=0).unsqueeze(0),
                                           self.W_column.weight[:, :self.rank] / torch.norm(self.W_column.weight[:, :self.rank], dim=0).unsqueeze(0)])

        return W

    def get_components(self, observation=False):
        return [self.construct_trial_weight_observation().T if observation else self.construct_trial_weight().T,
                self.W_row.weight.T,
                self.W_column.weight.T]
