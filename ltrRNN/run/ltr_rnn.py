from ltrRNN.run import fit, cross_validation

import numpy as np
import pickle


def import_data_(data_file):
    '''
    Loads the data of a task-tained RNN of motor perturbation learning in order to fit an ltrRNN to it.

    :param data_file: file containing the data

    :return: rnn_activity: (ndarray) time x trial x neuron. The activity of the RNN over learning.
    :return: condition: (ndarray) trial. The condition i.e. the target number (integer).
    :return: times: (ndarray) time. The time within a trial.
    :return: epoch: (ndarray) 3. The trial at which the epoch changes.
    :return: trial_id: (ndarray) trial. The true id of a trial, in case some trials are discarded.
    '''

    with open(data_file, 'rb') as f:
      data = pickle.load(f)

    # ===== Loading specific to your dataset =====
    neural_data = data['rnn_activity'][:,:, 0].transpose(1, 0, 2)
    condition = data['condition'][:, 0]*8

    go_cue = data['additional_information']['preparatory_duration']+data['additional_information']['random_duration']/2
    times = data['time']-go_cue

    epoch = np.array(['BL' for i in range(data['epochs']['perturbation'])] +
                     ['AD' for i in range(data['epochs']['washout']-data['epochs']['perturbation'])] +
                     ['WO' for i in range(data['condition'].shape[0]-data['epochs']['washout'])])

    trial_id = np.arange(len(condition))
    # ===============================================

    return neural_data, condition, times, epoch, trial_id

def import_data(data_file):
    '''
    Loads the data of a task-tained RNN of motor perturbation learning in order to fit an ltrRNN to it.

    :param data_file: file containing the data

    :return: rnn_activity: (ndarray) time x trial x neuron. The activity of the RNN over learning.
    :return: condition: (ndarray) trial. The condition i.e. the target number (integer).
    :return: times: (ndarray) time. The time within a trial.
    :return: epoch: (ndarray) 3. The trial at which the epoch changes.
    :return: trial_id: (ndarray) trial. The true id of a trial, in case some trials are discarded.
    '''

    with open(data_file, 'rb') as f:
      data = pickle.load(f)

    # ===== Loading specific to your dataset =====
    neural_data = data['rnn_activity'][:,:, 0].transpose(1, 0, 2)
    condition = data['condition'][:, 0]*8

    go_cue = data['additional_information']['preparatory_duration']+data['additional_information']['random_duration']/2
    times = data['time']-go_cue

    epoch = np.concatenate([[i]*50 for i in range(5)])

    trial_id = np.arange(len(condition))
    # ===============================================

    return neural_data, condition, times, epoch, trial_id

if __name__ == '__main__':

    neural_data, condition, times, epoch, trial_id = import_data('/home/arthur/PycharmProjects/LtrRNN-github/example_data/task_trained_rnn_data.pkl')

    parameters = {
        'seed': 1,

        # LtrRNN hyperparameters
        'rnn_dim': 200,
        'rank': 5,

        'time_constant': 1.0,
        'noise': 0.0,  # The RNN is driven by a Wiener process (can be useful to regularize dynamics)
        'bias': False,

        'duration': 2,  # How long the RNN is simulated (e.g. real experiment seconds)

        # Decoder -- the linear map from the RNN dimension to the data neuron dimension
        'orthogonal_decoder': True,  # Orthonormal map (up to scaling)
        'decoder_bias': False,  # Allows for translation

        # Gaussian kernel
        'sigma_observation': 0.1,  # Whether to allow non-smooth trial-to-trial variability
        'l': 15,  # The length of the kernel, acts like a time (trial) constant
        'sigma': 0.1,  # Magnitude of trial to trial variability
        'discontinuous_covariance': False,  # Requires epoch parameter if True

        # Controls (input) hyperparameters
        # If you have a preparatory period this will allow having trial-to-trial variability in the initial (fit to data) state
        'preparatory_steps': 20,
        'control_preparatory': True,
        'control_execution': False,
        'fit_preparatory': False,
        'condition_specific_init': False,
        'in_space_control': True,  # whether to constrain the controls to the span of a_i's

        'control_dim': 3,  # Ignored if in_space_control
        'control_dnn_dim': [150, 150],
        'control_hidden_dim': 150,
        'time_constant_control': 2,
        'trial_rank_control': -1,  # if -1 condition-wise

        # training hyperparameters
        'fraction_masked': 0.2,
        'fraction_test': 0.2,

        'learning_rate': 0.001,
        'test_freq': 50,
        'regularization': 0.01,
        'regularization_function': 'square',

        # If std of the test loss over the past steps_std_convergence is less than min_std_convergence, break
        'training_iterations': 3000,
        'min_std_convergence': 0.001,
        'steps_std_convergence': 200,

        'optimize': True
    }

    print(neural_data.shape, condition.shape, times.shape, epoch.shape, trial_id.shape)

    fit(parameters, neural_data)
