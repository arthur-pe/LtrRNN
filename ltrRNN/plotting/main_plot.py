from ltrRNN.plotting import plot_2d, plot_3d, utils

import numpy as np
import torch
import matplotlib.pyplot as plt


def get_figure(rows, columns, directory):
    """
    Builds a #rows x #columns numpy array of matplotlib axes.
    """

    fig = plt.figure(figsize=(columns * 4, rows * 4), constrained_layout=True, dpi=60)
    gs = fig.add_gridspec(ncols=columns, nrows=rows)
    axs_3d = []
    axs_ignored = []
    for i in range(0, 2):
        for j in range(2, 4):
            axs_ignored.append([i, j])
    axs_3d = axs_3d + [[2, 0], [2, 1]]
    axs = np.array([[fig.add_subplot(gs[i, j], projection=('3d' if [j, i] in axs_3d else None)) if [j, i] not in axs_ignored else None
                     for i in range(rows)] for j in range(columns)])

    ax_giga = fig.add_subplot(gs[2:4, 0:2], projection='3d')

    fig.suptitle('Dir:' + directory.split('/')[-1])

    return fig, axs, ax_giga


# Note: not centered
def var_exp_per_pc(x, number_pc):
    """
    Fraction variance explained per principal component (non-centered). The variance is not element-wise.
    """

    U, S, V = np.linalg.svd(x.reshape(-1, x.shape[-1]), full_matrices=False)

    x_projected = x @ V.T[:, :number_pc]
    temp_x = x_projected @ V[:number_pc]

    return np.mean(np.square(temp_x-x))/np.var(x)


def giga_projection(ax_giga, rnn_mp, x_projected, normalized_condition, cmap_per_condition, cmap, grad,
                    max_time, min_trial, max_trial, steps, times):
    """
    Projection on the columns of the tensor rank parameterization.
    """

    title = 'Projection on $\mathbf{a}_{1:3}$'
    ax_giga.set_title(title)

    trials_dimension = list(rnn_mp.shape)[1]

    plot_3d.trajectories_gradient_shadow_per_condition(ax_giga,
                                                       x_projected[:max_time, min_trial:max_trial:steps],
                                                       normalized_condition[min_trial:max_trial:steps],
                                                       cmap_per_condition,
                                                       condition_gradient=grad[min_trial:max_trial:steps],
                                                       linewidth=2.0, alpha_shadow=0.2)

    go_cue = np.argmax(times>=0)
    facecolors = np.ones((trials_dimension, 4))
    ax_giga.scatter(x_projected[go_cue, :max_trial:steps, 0],
                    x_projected[go_cue, :max_trial:steps, 1],
                    x_projected[go_cue, :max_trial:steps, 2],
                    color=cmap(normalized_condition[:max_trial:steps])[:, :3],
                    facecolor=facecolors[:max_trial:steps],
                    s=20,
                    linewidth=2.0)


def projection_over_time(axs, x_projected, condition_np, normalized_condition, grad, steps, ts_experiment,
                         cmap_per_condition, parameters):

    axs[0].set_title('Column factors projection')
    for i in range(min(len(axs), parameters['rank'])):
        for j, c in enumerate(np.unique(condition_np)):
            x_temp = x_projected[:, normalized_condition == c]
            grad_temp = grad[normalized_condition == c]
            for k in range(0, x_temp.shape[1], steps):
                axs[i].plot(ts_experiment, x_temp[:, k, i], color=cmap_per_condition[j](grad_temp[k]), alpha=0.3)

        utils.set_bottom_axis(axs[i])
        axs[i].set_xlabel('Time (ms)')
        ylabel = 'Projection on '
        if i < parameters['rank']:
            ylabel += '$\mathbf{a}_{' + str(i + 1) + '}$'
        else:
            ylabel += '$B_{' + str(parameters['rank'] - i + 1) + '}$'
        axs[i].set_ylabel(ylabel)


def data_pca(ax, full_neural_data, normalized_condition, cmap_per_condition, grad, min_trial, max_trial, steps):
    U, S, V = torch.pca_lowrank(full_neural_data.reshape(-1, list(full_neural_data.shape)[-1]), q=3)
    V = -V.detach().cpu().numpy().T
    ax.set_title('PCA data')
    full_neural_data = full_neural_data - full_neural_data.mean(dim=(0, 1))
    plot_3d.trajectories_gradient_shadow_per_condition(ax,
                                                       full_neural_data[:,
                                                       min_trial:max_trial:steps].detach().cpu().numpy() @ V.T,
                                                       normalized_condition[min_trial:max_trial:steps],
                                                       cmap_per_condition,
                                                       condition_gradient=grad[min_trial:max_trial:steps])


def sorted_by_peak(ax, rnn_activity, ts_experiment):

    trial_number = 10
    ax.set_title('Sorted by peak act, trial:' + str(trial_number))
    ax.set_xlabel('Time (ms)'), ax.set_ylabel('Neuron')
    plot_2d.sorted_activity(ax, rnn_activity[:, trial_number], t=ts_experiment)


def single_neurons(ax, ts_experiment, rnn_activity, normalized_condition, cmap_per_condition, trials=50):
    ax.set_title('Two neurons, trial:gradient')
    trials = min(trials, list(rnn_activity.shape)[1])
    steps = int(list(rnn_activity.shape)[1] / trials)
    plot_2d.trajectories_over_time_per_condition(ax, ts_experiment, rnn_activity[:, ::steps, :2],
                                                 normalized_condition[::steps],
                                                 cmap_per_condition=cmap_per_condition, alpha=0.5)

    ax.set_ylim(-1, 1)
    ax.set_xlabel('Time (ms)')


def controls_over_time(ax, ts_experiment, controls, normalized_condition, ts_experiment_prep, cmap, parameters):
    
    if parameters['control_preparatory'] or parameters['control_execution']:
        ax.set_title(str(parameters['control_dim']) + 'D controls per condition')
        plot_2d.controls(ax, ts_experiment if parameters['control_execution'] else ts_experiment_prep, controls,
                         normalized_condition, cmap=cmap)

    ax.set_ylim(-1.05, 1.05), ax.set_xlabel('Time (ms)')


def trial_factors(axs, components, components_latent, normalized_condition, trial_ids_np, cmap, epoch, epoch_starts, epoch_labels,
                  cmap_epoch, parameters, trial_ids):

    axs[0].set_title('Trial factors')
    plot_2d.factors(axs, components[0].detach().cpu().numpy(), normalized_condition,
                    indices=trial_ids_np, set_lim=True,
                    cmap=cmap, sort=False, zorder=4)

    temp = epoch_starts + [len(epoch)-1]
    for i in axs:
        for ei in range(len(epoch_starts)):
            i.axvspan(trial_ids_np[temp[ei]], trial_ids_np[temp[ei+1]],
                      facecolor=cmap_epoch(ei), label=epoch_labels[ei], zorder=1, alpha=0.2)
            i.set_xlabel('Trial')

    trial_components_latent = components_latent[0].detach().cpu().numpy()

    for i in range(min(len(axs), parameters['rank'])):
        temp = trial_components_latent[i]
        for e in np.unique(epoch):
            axs[i].plot(trial_ids_np[epoch == e], temp[epoch == e], color=(0.3, 0.3, 0.3),
                           linewidth=2.5, zorder=5)
        axs[i].fill_between(trial_ids.cpu().numpy(),
                               temp - 2 * parameters['sigma_observation'],
                               temp + 2 * parameters['sigma_observation'],
                               color=(0.9, 0.3, 1.0, 0.1), zorder=2)

    axs[min(parameters['rank'], len(axs) - 1)].legend()


def trial_factors_3d(ax, trial_components_latent, cmaps_trial_factor_3d, epoch, epoch_labels, parameters):

    ax.set_title('Trial factors in 3D')

    if parameters['rank'] >= 3:
        grads = [np.linspace(0.1, 1, np.sum(epoch == e)) for e in epoch_labels]
        colors = np.concatenate([cm(g) for cm, g in zip(cmaps_trial_factor_3d, grads)])[:, :3]

        ax.scatter(trial_components_latent[0, :], trial_components_latent[1, :], trial_components_latent[2, :],
                   color=colors, alpha=0.5, s=50)
        ax.view_init(elev=22, azim=-45, roll=0.5)
        ax.set_xlabel('$c_1$'), ax.set_ylabel('$c_2$'), ax.set_zlabel('$c_3$')


def loss(ax, neural_data, losses_array, parameters):

    ax.set_xlabel('Iteration'), ax.set_ylabel('MSE')
    neural_data_np = neural_data.cpu().numpy()
    ax.axhline(var_exp_per_pc(neural_data_np, parameters['rank']),
                      linestyle='--', color='grey',
                      label=str(parameters['rank']) + ' PCs')
    ax.plot(losses_array[:, 0], color='black', label='Train')
    ax.plot(losses_array[:, 1], color='red', label='Test')
    ax.set_title('Loss')
    utils.set_bottom_axis(ax)
    ax.legend()


def eigenspectrum(ax, W_obs, epoch_id, cmaps_trial_factor_3d):
    epoch_id_np = epoch_id
    ax.set_title('Eigenspectrum $W^{(i)}$')
    plot_2d.eigenspectrum_per_condition(ax, W_obs[::5],
                                        cmap_per_condition=cmaps_trial_factor_3d,
                                        condition=epoch_id_np[::5], s=5)


def initial_condition(ax, xs_exec_cut, epoch, cmap, normalized_condition, epoch_labels):

    utils.remove_axes(ax)
    initial_state = xs_exec_cut[0][0]
    initial_state = initial_state.numpy(force=True)

    initial_state = initial_state - initial_state[epoch == epoch[0]].mean(axis=0)

    U, S, V = np.linalg.svd(initial_state[epoch == epoch[0]], full_matrices=False)
    colors = cmap(normalized_condition)
    colors[:, 3] = 0.3
    initial_state_projected = initial_state @ V.T
    unique_condition = np.unique(normalized_condition)

    markers = ['o', 'D', 's', 'v', '^', '<', '>', 'p', 'h']
    for ei, e in enumerate(np.unique(epoch)):
        temp = initial_state_projected[epoch == e]
        ax.scatter(temp[:, 0], temp[:, 1],  # temp[:,2],
                   color=colors[epoch == e], s=30, edgecolor=(1, 1, 1, 0), zorder=5, marker=markers[ei])

        condition_sample = np.array(
            [np.median(initial_state_projected[(normalized_condition == c) & (epoch == e)], axis=0) for c in
             unique_condition])
        ax.scatter(condition_sample[:, 0], condition_sample[:, 1],  # condition_sample[:, 2],
                   color=cmap(unique_condition), s=200 if ei == 0 else 100,
                   edgecolor=(1, 1, 1, 1),
                   zorder=7, marker=markers[ei], label=epoch_labels[ei])

        if ei == 0:
            condition_sample = np.concatenate([condition_sample, condition_sample[0:1]], axis=0)
            ax.plot(condition_sample[:, 0], condition_sample[:, 1],  # condition_sample[:, 2],
                    color=(0.4, 0.4, 0.4), zorder=6)
    ax.legend()
    ax.set_aspect('equal')
    ax.set_title('Initial state')


def vector_fields(axs_temp, net, projection, cmap_epoch, components, epoch_starts, epoch_labels, trial_dimension, device, parameters):

    c_vector_a = (1.0, 0.3, 0.4, 1.0)
    c_vector_b = (1.0, 0.8, 0.3, 1.0)

    bound = 2
    xxs = np.linspace(-bound, bound, 9)
    xxs_grid = np.stack(np.meshgrid(xxs, xxs), axis=-1)
    b_projection = components[1].numpy(force=True)[:parameters['rank']] @ projection.T

    def temp_f(x, i): return net.activation(x) @ net.W[i] - x

    for axi in range(min(2, int(parameters['rank'] / 2))):
        ax = axs_temp[axi]
        utils.remove_axes(ax)
        xxs_grid_pad = np.concatenate([np.zeros(list(xxs_grid.shape[:2]) + [axi * 2]),
                                       xxs_grid,
                                       np.zeros(list(xxs_grid.shape[:2]) + [len(projection) - 2 * (axi + 1)])], axis=-1)

        temp = torch.tensor(xxs_grid_pad @ projection, device=device, dtype=torch.float)

        temp2 = epoch_starts+[trial_dimension-1]
        for ei in range(len(epoch_starts)):
            vf = temp_f(temp, int((temp2[ei]+temp2[ei+1]) / 2)).numpy(force=True) @ projection.T

            ax.quiver(xxs, xxs, vf[..., axi * 2], vf[..., axi * 2 + 1], color=cmap_epoch(ei), label=epoch_labels[ei], zorder=4,
                      angles='xy', scale_units='xy', scale=2)

        ax.quiver([0, 0], [0, 0], [1.06, 0], [0, 1.06], color=c_vector_a, label='$\mathbf{a}_i$', angles='xy',
                  scale_units='xy', scale=1, zorder=6, units='xy', width=0.08)
        ax.quiver(np.zeros(2), np.zeros(2), b_projection[axi * 2:axi * 2 + 2, axi * 2],
                  b_projection[axi * 2:axi * 2 + 2, axi * 2 + 1], color=c_vector_b, label='$\mathbf{b}_i$', angles='xy',
                  scale_units='xy', scale=1, zorder=7, units='xy', width=0.06)

        ax.legend(loc=3)


def text(ax, parameters):

    important_parameters = ['rank', 'rnn_dim', 'orthogonal_decoder', 'decoder_bias',
                            'regularization', 'seed']
    text = ''.join([k + ': ' + str(parameters[k]) + '\n' for k in important_parameters])
    ax.text(0, 0, text, fontsize='medium', va='bottom', ha='left')
    ax.axis('off')
