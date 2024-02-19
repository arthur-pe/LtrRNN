import matplotlib

from ltrRNN.rnn_architectures.tensor_rank import *
from ltrRNN.helper import control, classes, functions
from ltrRNN.plotting import utils
from ltrRNN.sde.systems import SDE
from ltrRNN.plotting import main_plot

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
from tqdm.auto import tqdm


def training_loop(sde_preparatory, sde_execution, net, condition_wise_map, rnn_to_data, neural_data,
          train_mask, test_mask, parameters, directory,
          epoch, # (trial)
          epoch_starts,  # (#epochs + 1) [0, ..., trial_dim]
          epoch_labels, # sorted by appearance in epoch
          optim, trials_dimension, trial_ids, cmap, ts_execution, ts_preparatory, ts_experiment, normalized_condition,
          cmap_per_condition, device):

    rows, columns = 5, 5
    fig, axs, ax_giga = main_plot.get_figure(rows, columns, directory)

    matplotlib.rcParams['font.size'] = 14

    losses = []

    iterator = tqdm(range(parameters['training_iterations']), desc="Training initialization ... ", unit=" iteration")

    for training_iteration in iterator:

        sde_preparatory.build_parameterization()

        # ========= Evaluation =========
        xs_prep = functions.sdeint_aaeh(sde_preparatory, sde_preparatory.get_initial_state(len(epoch)), ts_preparatory)
        xs_prep_cut = sde_preparatory.cut_states(xs_prep)
        if parameters['control_execution']:
            xs_exec = functions.sdeint_aaeh(sde_execution, xs_prep[-1], ts_execution)
        else:
            xs_exec = functions.sdeint_aaeh(sde_execution, xs_prep_cut[0][-1], ts_execution)

        xs_exec_cut = sde_execution.cut_states(xs_exec)

        rnn_mp = torch.cat([xs_prep_cut[0], xs_exec_cut[0][1:]])

        data_estimate = rnn_to_data(net.activation(rnn_mp if parameters['fit_preparatory'] else xs_exec_cut[0]))

        # ========= Loss =========
        mask_trial = (torch.rand(trials_dimension, device=device) < (1 - parameters['fraction_masked']))
        full_mask = train_mask & mask_trial.unsqueeze(0).unsqueeze(-1)

        l = torch.mean(((data_estimate - neural_data)[full_mask])**2)
        l_test = torch.mean(((data_estimate - neural_data)[test_mask])**2)
        l_total = torch.mean(((data_estimate - neural_data)[train_mask])**2)

        # ========= Regularization =========
        reg_fn = torch.abs if parameters['regularization_function'] == 'abs' else torch.square

        l_reg_rnn = reg_fn(rnn_mp[:, mask_trial]).mean()
        l_reg_control_prep = reg_fn(condition_wise_map.h(xs_prep_cut[1]))[:, mask_trial].mean()
        l_reg_control_exec = reg_fn(condition_wise_map.h(xs_exec_cut[1]))[:, mask_trial].mean() if parameters['control_execution'] else torch.tensor(0)

        l_reg_total = l_reg_control_prep + l_reg_control_exec + l_reg_rnn

        l += l_reg_total * parameters['regularization']

        losses.append([l_total.item(), l_test.item()])

        if training_iteration % parameters['test_freq'] == 0:
            with torch.no_grad():

                iterator.set_description(f'Iteration{training_iteration}, l_total: {l_total.item()}, l_test:{l_test.item()}, l_reg: {l_reg_total.item()},\
                      l_reg_control_prep:, {l_reg_control_prep.item()}, l_reg_control_exec: {l_reg_control_exec.item()},\
                      l_reg_rnn: {l_reg_rnn.item()}')

                for i in axs:
                    for j in i:
                        if j is not None: j.cla()
                ax_giga.cla()

                # ========= Arrays to plot ==========
                ts_experiment_prep = ts_experiment[:np.argmax(ts_experiment >= 0) +1 ]

                rnn_activity = net.activation(rnn_mp).numpy(force=True)
                rnn_mp = rnn_mp.numpy(force=True)

                controls = [xs_prep_cut[1]]
                if parameters['control_execution']: controls.append(xs_exec_cut[1][1:])
                controls = condition_wise_map(torch.cat(controls)).numpy(force=True)

                steps = 1
                min_trial = 0
                max_trial = 150
                max_time = len(rnn_mp)

                cmap_epoch = matplotlib.colormaps['Set2']

                cmaps_trial_factor_3d = [utils.get_cmap_interpolated(c * 0.1 + 0.9, c) for c in
                                         [np.array(cmap_epoch(ei))[:3] for ei in range(len(epoch_labels))]]

                # ========= Projection ==========
                components = net.get_components(observation=True)

                projection = components[2].numpy(force=True)
                projection = projection / np.linalg.norm(projection, axis=1)[:, np.newaxis]
                projection = projection[:parameters['rank']]

                temp = epoch_starts + [trials_dimension]
                grad = np.concatenate([np.linspace(0.1, 1.0, temp[i+1] - temp[i]) for i in range(len(epoch_starts))])

                x_projected = rnn_mp @ projection.T

                # =========== Plotting ===========

                # Big plot
                if parameters['rank'] >= 3:
                    main_plot.giga_projection(ax_giga, rnn_mp, x_projected, normalized_condition, cmap_per_condition, cmap, grad,
                                          max_time, min_trial, max_trial, steps, ts_experiment)

                # Projection over time
                main_plot.projection_over_time(axs[4], x_projected, normalized_condition, normalized_condition, grad, steps,
                                               ts_experiment, cmap_per_condition, parameters)

                # PCA on data
                main_plot.data_pca(axs[2][0], neural_data, normalized_condition, cmap_per_condition, grad,
                                   min_trial, max_trial, steps)

                # Sorted by peak
                main_plot.sorted_by_peak(axs[0][0], rnn_activity, ts_experiment)

                # Single neuron
                main_plot.single_neurons(axs[1][0], ts_experiment, rnn_activity, normalized_condition, cmap_per_condition)

                # Controls
                main_plot.controls_over_time(axs[0][1], ts_experiment, controls, normalized_condition, ts_experiment_prep, cmap, parameters)

                # Trial factors
                components_latent = net.get_components(observation=False)
                main_plot.trial_factors(axs[3], components, components_latent, normalized_condition,
                                        trial_ids.numpy(force=True), cmap, epoch, epoch_starts, epoch_labels, cmap_epoch,
                                        parameters, trial_ids)

                # Trial factors in 3D
                trial_components_latent = components_latent[0].detach().cpu().numpy()
                main_plot.trial_factors_3d(axs[2][1], trial_components_latent, cmaps_trial_factor_3d,
                                           epoch, epoch_labels, parameters)

                # Loss
                losses_array = np.array(losses) / np.var(neural_data.numpy(force=True))
                main_plot.loss(axs[2][3], neural_data, losses_array, parameters)

                # Eigenspectrum
                W_obs = net.construct_weight(observation=False).numpy(force=True)
                main_plot.eigenspectrum(axs[2][2], W_obs, epoch, cmaps_trial_factor_3d)

                # Initial condition
                main_plot.initial_condition(axs[1][1], xs_exec_cut, epoch, cmap, normalized_condition, epoch_labels)

                # Vector fields
                axs_temp = [axs[1, 4], axs[2, 4]]
                main_plot.vector_fields(axs_temp, net, projection, cmap_epoch, components, epoch_starts, epoch_labels,
                                        len(epoch), device, parameters)

                # Text
                ax = axs[0, 4]
                main_plot.text(ax, parameters)

                plt.draw()
                plt.pause(5) # Increase for some CPU configs

                plt.savefig(directory + '/' + directory.split('/')[-1] + '.pdf')

                if training_iteration == 0: plt.savefig(directory + '/' + directory.split('/')[-1] + '-0.pdf')

                torch.save(sde_preparatory.state_dict(), directory + '/model.pt')
                torch.save(rnn_to_data.state_dict(), directory + '/map.pt')

                # Criteria to stop optim
                if len(losses_array )>=parameters['steps_std_convergence']:
                    print(losses_array[-parameters['steps_std_convergence']:, 1].std())

        if len(losses_array )>=parameters['steps_std_convergence']:
            if losses_array[-parameters['steps_std_convergence']: ,1].std( ) <parameters['min_std_convergence']:
                break

        if parameters['optimize']:
            optim.zero_grad()
            l.backward()
            sde_preparatory.backward_parameterization()
            optim.step()

    plt.close(fig)

    return losses_array[-parameters['steps_std_convergence']:].mean(axis=0)


def train(parameters, neural_data, condition, times, epoch, trial_ids, train_mask, test_mask,
         directory, load_directory='.',
         device=('cuda' if torch.cuda.is_available() else 'cpu')):

    torch.manual_seed(parameters['seed'])
    np.random.seed(parameters['seed'])

    condition = np.zeros(neural_data.shape[1]) if condition is None else condition
    times = np.linspace(0, 1, neural_data.shape[0]) if times is None else times
    epoch = np.zeros(neural_data.shape[1]) if epoch is None else epoch
    trial_ids = np.arange(neural_data.shape[1]) if trial_ids is None else trial_ids
    train_mask = torch.ones(list(neural_data.shape)) if train_mask is None else train_mask
    test_mask = torch.ones(list(neural_data.shape)) if test_mask is None else test_mask

    prep, start, stop = 0, np.argmax(times >= 0), len(times)
    ts_experiment = times[prep:stop]*100

    neural_data = neural_data[start:stop] if not parameters['fit_preparatory'] else neural_data[prep:stop]
    time_dimension, trials_dimension, neurons_dimension = neural_data.shape
    condition_dimension = len(np.unique(condition))

    epoch_labels = epoch[np.sort(np.unique(epoch, return_index=True)[1])]
    epoch_starts = [np.argmax(epoch == e) for e in epoch_labels]

    print('Dimensions - Time:', time_dimension, ' Trial:', trials_dimension, 'Neuron:', neurons_dimension, 'Condition:', condition_dimension)

    normalized_condition = condition / (np.max(condition ) + 1)
    neural_data, condition, trial_ids = torch.tensor(neural_data, device=device), torch.tensor(condition, device=device, dtype=torch.long), torch.tensor \
        (trial_ids, device=device)

    cmap = matplotlib.colormaps['hsv']
    cmap_per_condition = [utils.get_cmap_black(cmap(i)) for i in np.unique(normalized_condition)]

    activation = torch.tanh

    kernel = RationalQuadratic(parameters['l'], parameters['sigma'])

    if parameters['in_space_control']: parameters['control_dim'] = parameters['rank']

    net = TensorRankSmooth(parameters['rnn_dim'], parameters['rank'],
                           trial_ids, None, kernel, epoch=epoch if parameters['discontinuous_covariance'] else None,
                           activation=activation, std_observation=parameters['sigma_observation'],
                           optimize_input_maps=True, in_space_inputs=parameters['in_space_control'],
                           condition=condition if parameters['condition_specific_init'] else None,
                           bias=parameters['bias'],
                           in_dims=(parameters['control_dim'],),
                           time_constant=parameters['time_constant'],
                           noise=parameters['noise'], device=device)

    activation_control = torch.relu
    controller = control.TrialRankController(parameters['control_hidden_dim'], trials_dimension, condition_dimension,
                                             parameters['control_dnn_dim'],
                                             time_constant=parameters['time_constant_control'],
                                             in_dims=[],
                                             init='optimized', device=device, activation=activation_control)

    condition_wise_map = classes.ConditionWiseControlMap(condition, parameters['control_hidden_dim'],
                                                         parameters['control_dim'], activation=activation,
                                                         device=device)

    graph = np.array([[1, condition_wise_map],
                      [0, 1]])
    sde_preparatory = SDE([net, controller], graph)

    if parameters['control_execution']:
        graph = np.array([[1, condition_wise_map],
                          [0, 1]])
        sde_execution = SDE([net, controller], graph)
    else:
        sde_execution = SDE([net], [[1]])

    if parameters['orthogonal_decoder']:
        rnn_to_data = classes.OrthogonalScaled(parameters['rnn_dim'], neurons_dimension,
                                               bias=parameters['decoder_bias'],
                                               scaling=False,
                                               scaling_neuron_wise=False, device=device)
    else:
        rnn_to_data = nn.Linear(parameters['rnn_dim'], neurons_dimension,
                                bias=parameters['decoder_bias'], device=device)

    param = set(list(sde_preparatory.parameters()) + list(rnn_to_data.parameters()))

    optim = torch.optim.Adam(param, lr=parameters['learning_rate'])

    ts = torch.linspace(0, parameters['duration'], stop - start)
    ts_preparatory = torch.linspace(0, (start - prep + 1) * (ts[1] - ts[0]), start - prep + 1)
    ts_execution = ts

    if load_directory != '.':
        sde_preparatory.load_state_dict(torch.load(load_directory + '/model.pt', map_location=device), strict=False)
        rnn_to_data.load_state_dict(torch.load(load_directory + '/map.pt', map_location=device))

    return training_loop(sde_preparatory, sde_execution, net, condition_wise_map, rnn_to_data,
                 neural_data,
                 train_mask, test_mask, parameters, directory, epoch, epoch_starts, epoch_labels, optim,
                 trials_dimension, trial_ids, cmap,
                 ts_execution, ts_preparatory, ts_experiment, normalized_condition, cmap_per_condition, device)

