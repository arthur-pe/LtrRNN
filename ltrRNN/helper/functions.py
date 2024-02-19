from torchsde import sdeint_adjoint
from torchsde import sdeint
from datetime import datetime
import os
import shutil
import yaml
import sys

import torch
import numpy as np
from typing import Iterable

# torchsde can recurse deep
sys.setrecursionlimit(10000)


# Convenience functions for torch sde sdeint
def sdeint_aaeh(*args, **kwargs):
    return sdeint(*args, **kwargs, adaptive=True, method='euler_heun')


def sdeint_aaeh_a(*args, **kwargs):
    return sdeint_adjoint(*args, **kwargs, adaptive=True, method='euler_heun',
                          adjoint_adaptive=True, adjoint_method='euler_heun')


def sdeint_aaeh_r(*args, **kwargs):
    return sdeint_adjoint(*args, **kwargs, adaptive=True, method='reversible_heun',
                            adjoint_adaptive=True, adjoint_method='adjoint_reversible_heun')


def make_directory(directory_name='runs', sub_directory=None):

    now = datetime.now()
    date_time = now.strftime("%d-%m-%Y_%H_%M_%S")

    directory = './'+directory_name+'/all/' + date_time

    if not os.path.exists(directory):
        os.makedirs(directory)

    if sub_directory is not None:
        for i in sub_directory:
            if not os.path.exists(directory+'/'+i):
                os.makedirs(directory+'/'+i)

    print('directory:', directory)

    return directory


def load_yaml(load_directory, directory, parameter_file='/parameters.yaml'):

    with open(load_directory+parameter_file, 'r') as f:
        parameters = yaml.safe_load(f)
    for i in parameters:
        print(i, ':', parameters[i])
    shutil.copyfile(load_directory+parameter_file, directory+parameter_file)

    return parameters


def tonp(x):

    return x.numpy(force=True)


def block_mask(dimensions: Iterable[int],
                train_blocks_dimensions: Iterable[int],
                test_blocks_dimensions: Iterable[int],
                fraction_test: float,
                exact: bool = True,
                device: str = ('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Builds train and test masks.
    The train mask has block of entries masked.
    The test mask has the opposite entries masked, plus the boundaries of the blocks.

    :param dimensions: Dimensions of the mask.
    :param train_blocks_dimensions: Dimensions of the blocks discarded for training will be 2*train_block_dimensions+1
    :param test_blocks_dimensions: Dimensions of the blocks retained for testing will be 2*test_block_dimensions+1
    :param fraction_test: The fraction of entries used for testing.
    :param exact:   If exact then the number of blocks will be number_blocks (slower).
                    If not exact, the number of blocks will be on average number_blocks (faster).
    :param device: torch device (e.g. 'cuda' or 'cpu').
    :return: train_mask, test_mask
    """

    valence = len(dimensions)

    flattened_max_dim = np.prod(dimensions)

    if not np.prod((np.array(train_blocks_dimensions)-np.array(test_blocks_dimensions))>=0):
        raise Exception('For all i it should be that train_blocks_dimensions[i]>=test_blocks_dimensions[i].')

    number_blocks = int(fraction_test * np.prod(np.array(dimensions)) / np.prod(1 + 2 * np.array(test_blocks_dimensions)))

    if exact:
        start = torch.zeros(flattened_max_dim, device=device)
        start[:number_blocks] = 1
        start = start[torch.randperm(flattened_max_dim, device=device)]
        start = start.reshape(dimensions)
    else:
        density = number_blocks / flattened_max_dim
        start = (torch.rand(tuple(dimensions), device=device) < density).long()

    start_index = start.nonzero()
    number_blocks = len(start_index)

    # Build outer-blocks mask
    a = [[slice(torch.clip(start_index[j][i]-train_blocks_dimensions[i],min=0, max=dimensions[i]),
                torch.clip(start_index[j][i]+train_blocks_dimensions[i]+1,min=0, max=dimensions[i]))
          for i in range(valence)] for j in range(number_blocks)]

    train_mask = torch.full(dimensions, True, device=device)

    for j in a: train_mask[j] = 0

    # Build inner-blocks tensor
    a = [[slice(torch.clip(start_index[j][i]-test_blocks_dimensions[i],min=0, max=dimensions[i]),
                torch.clip(start_index[j][i]+test_blocks_dimensions[i]+1,min=0, max=dimensions[i]))
                for i in range(valence)] for j in range(number_blocks)]

    test_mask = torch.full(dimensions, False, device=device)

    for j in a: test_mask[j] = 1

    return train_mask, test_mask
