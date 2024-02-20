from ltrRNN.helper import functions
from ltrRNN.plotting import utils
from .training import train

import torch
from matplotlib import pyplot as plt
import matplotlib
import numpy as np

from datetime import datetime
import os
import yaml


def grid_search(hyperparameters, neural_data, condition=None, times=None, epochs=None, trial_ids=None, train_mask=None, test_mask=None,
                     cv_hyperparameters={'rnn_dim': [50, 200], 'rank': [1, 2, 3, 4, 5, 10, 20]}, seeds=(1, 2, 3),
                     device=('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Cross-validate ltrRNN hyperparameters. The output is saved within a ./cv/runs/... directory.

    :param hyperparameters: Dictionary of hyperparameters (see example notebook).
    :param neural_data: Numpy array of shape (time x trial x neuron). Can either be firing rate estimates (e.g. z-scored), or spikes.
    :param condition: Numpy array of shape (trial) indicating the experimental condition of each trial (int).
    :param times: Numpy array of shape (time) indicating time (e.g. within the experiment).
    :param epochs: Numpy array of shape (trial) indicating the epoch of the task (e.g. perturbed/not perturbed).
    :param trial_ids: Numpy array of shape (trial) indicating the true trial id within the experiment (e.g. if trials were discarded).
    :param train_mask: Torch tensor of shape (time x trial x neuron) which indicates which entries to compute the gradient with respect to.
    :param test_mask: Torch tensor of shape (time x trial x neuron) which indicates which entries to compute the test loss with resepct to.
    :param cv_hyperparameters: the hyperparameters which should be modified over the cross-validation grid. Note: supports only 2 hyperparameters for now.
    :param seeds: The seeds to use for each combination of hyperparameters so that the total number of runs is #hyperparameter1 x #hyperparameter2 x #seeds.
    :param device: Torch device.
    """

    directory = functions.make_directory('cv')

    with open(directory+'/parameters.yaml', 'w') as f:
        yaml.dump(hyperparameters, f)

    keys = list(cv_hyperparameters.keys())

    cv_losses = np.full((len(cv_hyperparameters[keys[0]]), len(cv_hyperparameters[keys[1]]), len(seeds)), np.nan)
    fig = plt.figure(figsize=(5, 4), constrained_layout=True)
    ax = fig.add_subplot()
    cmap = matplotlib.colormaps['Set2']

    plt.show(block=False)

    for p1i, p1 in enumerate(cv_hyperparameters[keys[0]]):
        for p2i, p2 in enumerate(cv_hyperparameters[keys[1]]):
            for si, s in enumerate(seeds):
                print(keys[0], p1, keys[1], p2, 'seed', s)
                hyperparameters[keys[0]] = p1
                hyperparameters[keys[1]] = p2
                hyperparameters['seed'] = s

                now = datetime.now()
                date_time = now.strftime("%d-%m-%Y_%H_%M_%S")
                directory_run = directory + '/' + date_time
                if not os.path.exists(directory_run):
                    os.makedirs(directory_run)
                try:
                    l = train(hyperparameters, neural_data, condition, times, epochs, trial_ids, directory_run,
                              train_mask, test_mask, '.', device=device)[1]
                except Exception as e:
                    print('\n\n')
                    print(e)
                    print('\n\n')

                    l = np.nan
                cv_losses[p1i, p2i, si] = l

                np.save(directory+'/cv_grid.npy', cv_losses)

                cv_losses_mean = cv_losses.mean(axis=-1)
                cv_loss_std = cv_losses.std(axis=-1)

                utils.set_bottom_axis(ax)

                ax.cla()
                for pi, p in enumerate(cv_hyperparameters[keys[0]]):
                    ax.errorbar(cv_hyperparameters[keys[1]], cv_losses_mean[pi], cv_loss_std[pi], fmt='-o',
                                label=keys[0]+'='+str(p),color=cmap(pi), linewidth=1.5)

                ax.set_xticks(cv_hyperparameters[keys[1]])
                ax.set_xlabel(keys[1])
                ax.set_ylabel('MSE')
                ax.set_title('CV loss')
                ax.legend()

                plt.savefig(directory+'/cv_' + directory.split('/')[-1] +'.pdf')
                plt.draw()
                plt.pause(0.1)


def fit(hyperparameters,
        neural_data,
        condition=None,
        times=None,
        epochs=None,
        trial_ids=None,
        train_mask=None,
        test_mask=None,
        load_directory='.',
        device=('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Fits an ltrRNN to neural data. The output is saved within a ./runs/... directory.

    :param hyperparameters: Dictionary of hyperparameters (see example notebook).
    :param neural_data: Numpy array of shape (time x trial x neuron). Can either be firing rate estimates (e.g. z-scored), or spikes.
    :param condition: Numpy array of shape (trial) indicating the experimental condition of each trial (int).
    :param times: Numpy array of shape (time) indicating time (e.g. within the experiment).
    :param epochs: Numpy array of shape (trial) indicating the epoch of the task (e.g. perturbed/not perturbed).
    :param trial_ids: Numpy array of shape (trial) indicating the true trial id within the experiment (e.g. if trials were discarded).
    :param train_mask: Torch tensor of shape (time x trial x neuron) which indicates which entries to compute the gradient with respect to.
    :param test_mask: Torch tensor of shape (time x trial x neuron) which indicates which entries to compute the test loss with resepct to.
    :param load_directory: If training is stopped, indicates the directory which contains /model.pt.
    :param device: Torch device.
    """

    directory = functions.make_directory()

    with open(directory+'/parameters.yaml', 'w') as f:
        yaml.dump(hyperparameters, f)

    train(hyperparameters, neural_data, condition, times, epochs, trial_ids, train_mask, test_mask,
         directory, load_directory, device)
