from ltrRNN.plotting.utils import *
from sklearn.manifold import Isomap

from matplotlib import pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import scipy

default_linewidth = 2
box_widening = 0.1

def trajectories_gradient(ax, x, condition=None, cmap=default_cmap, alpha=1, linewidth=default_linewidth, gradient=None,
                          dashed=False, number_dashes=50, dash_density=0.7,
                          zorder=3, set_lim=True, widen_box=True):
    """
    :param x: (time, trial, 2)
    :param condition: (trial), in [0,1]
    """

    if isinstance(cmap, str):
        cmap = matplotlib.colormaps[cmap]

    if condition is None:
        condition = np.zeros(x.shape[1])

    if gradient is None:
        gradient = np.linspace(0.1,1,x.shape[0])

    cmaps = {i:get_cmap_white(cmap(i)) for i in np.unique(condition)}

    for i in range(x.shape[1]):
        plot_with_gradient_2d(ax, x[:,i,0], x[:,i,1],
                              gradient=gradient, cmap=cmaps[condition[i]],
                              dashed=dashed, number_dashes=number_dashes, dash_density=dash_density,
                              linewidth=linewidth, alpha=alpha, zorder=zorder, set_lim=set_lim)

    if widen_box:
        x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
        ax.set_xlim(ax.get_xlim()[0]-(x_lim[1]-x_lim[0])*box_widening, ax.get_xlim()[1]+(x_lim[1]-x_lim[0])*box_widening)
        ax.set_ylim(ax.get_ylim()[0]-(y_lim[1]-y_lim[0])*box_widening, ax.get_ylim()[1]+(y_lim[1]-y_lim[0])*box_widening)

    for side in ax.spines.keys():  # 'top', 'bottom', 'left', 'right'
        ax.spines[side].set_linewidth(axes_line_width)
        #ax.spines[side].set_color((0.2,0.2,0.2))
        ax.spines[side].set_capstyle('round')

    #ax.spines['right'].set_color((1, 1, 1, 0)), ax.spines['top'].set_color((1, 1, 1, 0))
    ax.tick_params(width=axes_line_width)


def trajectories_over_time(ax, t, x, condition=None, cmap=default_cmap, alpha=1, linewidth=default_linewidth, linestyle='-'):

    """
    :param x: (time, trial)
    :param t: (time)
    :param condition: (trial), in [0,1]
    """

    if isinstance(cmap, str):
        cmap = matplotlib.colormaps[cmap]

    if condition is None:
        condition = np.zeros(x.shape[1])

    for i in range(x.shape[1]):
        ax.plot(t, x[:,i], alpha=alpha, linewidth=linewidth, linestyle=linestyle, color=cmap(condition[i]))

    set_bottom_axis(ax)


def trajectories_per_condition(ax, x, condition, cmap_per_condition=None, alpha=1, linewidth=1.5, gradient=None,
                          dashed=False, number_dashes=50, dash_density=0.7,
                          zorder=3, widen_box=True):
    """
    :param x: (time, trial)
    :param condition: (trial), in [0,1]
    :param cmap_per_condition: (number_of_unique_conditions) in [0,1]
    """

    unique_condition = np.unique(condition)

    for i, u in enumerate(unique_condition):
        temp_x = x[:,condition==u]
        temp_cmap = cmap_per_condition[i] if cmap_per_condition is not None else default_cmap(u)
        trajectories_gradient(ax, temp_x, condition=np.linspace(0,1,temp_x.shape[1]),
                            cmap=temp_cmap, alpha=alpha, linewidth=linewidth,
                            gradient=gradient, dashed=dashed, number_dashes=number_dashes, dash_density=dash_density,
                            zorder=zorder, set_lim=True, widen_box=False)

    if widen_box:
        x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
        ax.set_xlim(ax.get_xlim()[0]-(x_lim[1]-x_lim[0])*box_widening, ax.get_xlim()[1]+(x_lim[1]-x_lim[0])*box_widening)
        ax.set_ylim(ax.get_ylim()[0]-(y_lim[1]-y_lim[0])*box_widening, ax.get_ylim()[1]+(y_lim[1]-y_lim[0])*box_widening)

    for side in ax.spines.keys():
        ax.spines[side].set_linewidth(axes_line_width)
        ax.spines[side].set_capstyle('round')

    ax.tick_params(width=axes_line_width)


def trajectories_over_time_per_condition(ax, t, x, condition, cmap_per_condition, alpha=1, linewidth=default_linewidth, linestyle='-'):
    """
    :param x: (time, trial)
    :param condition: (trial), in [0,1]
    :param cmap_per_condition: (number_of_unique_conditions) in [0,1]
    """

    unique_condition = np.unique(condition)

    for i, u in enumerate(unique_condition):
        temp_x = x[:,condition==u]
        temp_cmap = cmap_per_condition[i] if cmap_per_condition is not None else default_cmap(u)
        trajectories_over_time(ax, t, temp_x, condition=np.linspace(0,1,temp_x.shape[1]), cmap=temp_cmap,
                               alpha=alpha, linewidth=linewidth, linestyle=linestyle)

    for side in ax.spines.keys():  # 'top', 'bottom', 'left', 'right'
        ax.spines[side].set_linewidth(axes_line_width)
        #ax.spines[side].set_color((0.2,0.2,0.2))
        ax.spines[side].set_capstyle('round')

    #ax.spines['right'].set_color((1, 1, 1, 0)), ax.spines['top'].set_color((1, 1, 1, 0))
    ax.tick_params(width=axes_line_width)


def eigenspectrum(ax, W, s=5, color=(0.3,0.3,0.3)):

    L, V = np.linalg.eig(W)

    ax.scatter(L.real, L.imag, color=color, s=s)

    ax.text(ax.get_xlim()[1], (ax.get_ylim()[1]-ax.get_ylim()[0])/50, 'Re', ha='right', va='bottom')
    ax.text((ax.get_xlim()[1]-ax.get_xlim()[0])/50, ax.get_ylim()[1], 'Im', ha='left', va='top')

    set_centered_axes(ax)


def eigenspectrum_per_trial(ax, W, s=10, colors=None, cmap='autumn_r'):

    if isinstance(cmap, str):
        cmap = matplotlib.colormaps[cmap]

    Ls, V = np.linalg.eig(W)

    if colors is None:
        colors = np.linspace(0, 1, len(Ls))

    for i, L in enumerate(Ls):
        c = list(cmap(colors[i]))
        c[3] = 0.7
        if i != len(Ls)-1:
            ax.scatter(L.real, L.imag, color=c, s=s, edgecolor=(1,1,1,0))
        else:
            ax.scatter(L.real, L.imag, color=c, s=s, edgecolor=c, facecolor=(1,1,1,1))

    ax.text(ax.get_xlim()[1], (ax.get_ylim()[1] - ax.get_ylim()[0]) / 50, 'Re', ha='right', va='bottom')
    ax.text((ax.get_xlim()[1] - ax.get_xlim()[0]) / 50, ax.get_ylim()[1], 'Im', ha='left', va='top')

    set_centered_axes(ax)


def eigenspectrum_per_condition(ax, W, condition, cmap_per_condition, s=5, alpha=0.7, zorder=5, text=True, center=True):

    unique_condition = np.unique(condition)

    for i, u in enumerate(unique_condition):
        temp_W = W[condition==u]
        for j, w in enumerate(temp_W):
            L, V = np.linalg.eig(w)

            ax.scatter(L.real, L.imag, color=cmap_per_condition[i]((1+j)/len(temp_W)), s=s, alpha=alpha, zorder=zorder)

    if center:
        set_centered_axes(ax)

    if text:
        ax.text(ax.get_xlim()[1], (ax.get_ylim()[1]-ax.get_ylim()[0])/50, 'Re', ha='right', va='bottom')
        ax.text((ax.get_xlim()[1]-ax.get_xlim()[0])/50, ax.get_ylim()[1], 'Im', ha='left', va='top')


def sorted_activity(ax, x, t=None, cmap='RdBu_r', quantile=None):

    vmax = 1 if quantile is None else np.quantile(np.abs(x), quantile)

    indx = np.argsort(np.argmax(x, axis=0))
    temp = x.T[indx]
    if t is None:
        im = ax.matshow(temp, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')
    else:
        im = ax.matshow(temp, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto',
                        extent=[t[0],t[-1], 0, x.shape[-1]])
        ax.tick_params(bottom=True, top=False, labelbottom=True, labeltop=False)

    return im


def factors(axs, factors, condition, indices=None, sort=True, set_lim=True, cmap=default_cmap, alpha=1, s=10, zorder=10):

    cmap = matplotlib.cm.get_cmap(cmap)

    colors = np.array([cmap(i) for i in condition])

    indices = indices if indices is not None or True else np.arange(list(factors.shape[1]))

    if sort:
        idx = np.argsort(condition, kind='mergesort')
        colors = np.array([cmap(i) for i in condition[idx]])

    min_f, max_f = np.min(factors), np.max(factors)
    min_f, max_f = min_f-0.05*(max_f-min_f), max_f+0.05*(max_f-min_f)

    for i in range(min(len(axs), len(factors))):

        f = factors[i][idx] if sort else factors[i]

        axs[i].scatter(indices, f, c=colors, s=s, alpha=alpha, zorder=zorder)

        set_bottom_axis(axs[i])

        if set_lim: axs[i].set_ylim(min_f, max_f)


def controls(ax, t, control, condition, cmap=default_cmap, sample_size=1, linewidth=default_linewidth, alpha=1, linestyle='-'):
    cmap = matplotlib.cm.get_cmap(cmap)

    unique_condition = np.unique(condition)
    sample = np.concatenate([np.where(condition==i)[0][:sample_size] for i in unique_condition])
    for i in sample:
        for j in range(control.shape[-1]):
            ax.plot(t, control[:,i,j], color=cmap(condition[i]), linestyle=linestyle, linewidth=linewidth, alpha=alpha)

    set_bottom_axis(ax)

def nldr_per_condition(ax, x, condition, cmap_per_condition, method=None, alpha=0.7):

    if method is None:
        method = Isomap()

    embedded_x = method.fit_transform(x)

    for c_id, c in enumerate(np.unique(condition)):
        temp_embedded_x = embedded_x[condition==c]
        for x_id, x_e in enumerate(temp_embedded_x):
            ax.scatter(x_e[...,0], x_e[...,1],
                            color=cmap_per_condition[c_id](x_id/len(temp_embedded_x)), alpha=alpha, s=10, zorder=5)


def var_exp_per_pc(ax, x, y=None, max_pc=10, color='black', alpha=1, label=None):
    """
    :param ax:
    :param x: to project
    :param y: to compute PC
    """

    if y is None: y = x

    U, S, V = np.linalg.svd(y.reshape(-1, y.shape[-1]), full_matrices=False)

    x_projected =  x @ V.T[:,:max_pc]

    var_exp = []

    for i in range(1, max_pc+1):

        temp_x = x_projected[..., :i] @ V[:i]

        R = 1 - np.mean(np.square(x-temp_x))/np.var(x)
        #R = np.mean(np.square(x-temp_x))

        var_exp.append(R)

    ax.plot(np.arange(1, max_pc+1), var_exp, '-o', color=color, alpha=alpha, markerfacecolor=(1,1,1,1), label=label)

    return var_exp
