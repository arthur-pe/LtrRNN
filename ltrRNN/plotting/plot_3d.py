from ltrRNN.plotting.utils import *

import numpy as np
from matplotlib import pyplot as plt
import matplotlib

box_widening = 0.1


@decorator_set_pannels_3d
def trajectories_gradient(ax, x, condition=None, cmap=default_cmap, alpha=1, linewidth=1.5, gradient=None,
                          dashed=False, number_dashes=50, dash_density=0.7,
                          zorder=3, set_lim=True, widen_box=True):
    """
    :param x: (time, trial, 3)
    :param condition: (trial), in [0,1]
    """

    if widen_box:
        ax.set_xlim(10**6, -10**6), ax.set_ylim(10**6, -10**6), ax.set_zlim(10**6, -10**6)

    if isinstance(cmap, str):
        cmap = matplotlib.colormaps[cmap]

    if condition is None:
        condition = np.zeros(x.shape[1])

    if gradient is None:
        gradient = np.linspace(0.1,1,x.shape[0])

    cmaps = {i:get_cmap_white(cmap(i)) for i in np.unique(condition)}

    for i in range(x.shape[1]):
        plot_with_gradient_3d(ax, x[:,i,0], x[:,i,1], x[:,i,2],
                              gradient=gradient, cmap=cmaps[condition[i]],
                              dashed=dashed, number_dashes=number_dashes, dash_density=dash_density,
                              linewidth=linewidth, alpha=alpha, zorder=zorder, set_lim=set_lim)

    if widen_box:
        x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
        ax.set_xlim(ax.get_xlim()[0]-(x_lim[1]-x_lim[0])*box_widening, ax.get_xlim()[1]+(x_lim[1]-x_lim[0])*box_widening)
        ax.set_ylim(ax.get_ylim()[0]-(y_lim[1]-y_lim[0])*box_widening, ax.get_ylim()[1]+(y_lim[1]-y_lim[0])*box_widening)

    ax.view_init(18, -55, roll=1.35)

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.line.set_linewidth(axes_line_width)
        axis.line.set_solid_capstyle('round')


def trajectories_gradient_shadow(ax, x, condition=None, cmap=default_cmap, alpha=1, linewidth=1.5, gradient=None,
                          dashed=False, number_dashes=50, dash_density=0.7,
                          zorder=3, set_lim=True, widen_box=True, z_projection_height=0.5, alpha_shadow=0.1, zlim=None):
    """
    White to color trajectories

    :param x: (time, trial, 3)
    :param condition: (trial), in [0,1]
    """
    if widen_box:
        ax.set_xlim(10**6, -10**6), ax.set_ylim(10**6, -10**6), ax.set_zlim(10**6, -10**6)

    if isinstance(cmap, str):
        cmap = matplotlib.colormaps[cmap]

    if condition is None:
        condition = np.zeros(x.shape[1])

    if gradient is None:
        gradient = np.linspace(0.1,1,x.shape[0])

    trajectories_gradient(ax, x, condition, cmap, alpha, linewidth, gradient, dashed, number_dashes, dash_density,
                          zorder, set_lim, widen_box=False)

    if zlim is None:
        zlim = ax.get_zlim()[0] - z_projection_height * (ax.get_zlim()[1] - ax.get_zlim()[0])

    zeros = np.zeros(x.shape[0])+zlim

    cmaps = {i:get_cmap_white(cmap(i)) for i in np.unique(condition)}

    for i in range(x.shape[1]):
        plot_with_gradient_3d(ax, x[:, i, 0], x[:, i, 1], zeros,
                              gradient=gradient, cmap=cmaps[condition[i]],
                              dashed=dashed, number_dashes=number_dashes, dash_density=dash_density,
                              linewidth=linewidth, alpha=alpha_shadow, zorder=zorder, set_lim=set_lim)

    ax.set_box_aspect(aspect=(1, 1, 1.1))

    if widen_box:
        x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
        ax.set_xlim(ax.get_xlim()[0]-(x_lim[1]-x_lim[0])*box_widening, ax.get_xlim()[1]+(x_lim[1]-x_lim[0])*box_widening)
        ax.set_ylim(ax.get_ylim()[0]-(y_lim[1]-y_lim[0])*box_widening, ax.get_ylim()[1]+(y_lim[1]-y_lim[0])*box_widening)

def trajectories_gradient_shadow_per_condition(ax, x, condition, cmap_per_condition=None, alpha=1, linewidth=1.5,
                        gradient=None, condition_gradient=None, dashed=False, number_dashes=50, dash_density=0.7,
                        zorder=3, widen_box=True, z_projection_height=0.5, alpha_shadow=0.1):
    """
    :param x: (time, trial)
    :param condition: (trial), in [0,1]
    :param cmap_per_condition: (number_of_unique_conditions) in [0,1]
    """

    ax.set_xlim(10**6, -10**6), ax.set_ylim(10**6, -10**6), ax.set_zlim(10**6, -10**6)

    unique_condition = np.unique(condition)

    zlim = np.min(x[:,:,2]) - z_projection_height*(np.max(x[:,:,2])-np.min(x[:,:,2]))

    condition_gradient = np.linspace(0,1, x.shape[1]) if condition_gradient is None else condition_gradient

    for i, u in enumerate(unique_condition):
        temp_x = x[:,condition==u]
        temp_cmap = cmap_per_condition[i] if cmap_per_condition is not None else default_cmap(u)
        trajectories_gradient_shadow(ax, temp_x, condition=condition_gradient[condition==u],
                            cmap=temp_cmap, alpha=alpha, linewidth=linewidth,
                            gradient=gradient, dashed=dashed, number_dashes=number_dashes, dash_density=dash_density,
                            zorder=zorder, set_lim=True, widen_box=False, z_projection_height=z_projection_height,
                            alpha_shadow=alpha_shadow, zlim=zlim)

    if widen_box:
        x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
        ax.set_xlim(ax.get_xlim()[0]-(x_lim[1]-x_lim[0])*box_widening, ax.get_xlim()[1]+(x_lim[1]-x_lim[0])*box_widening)
        ax.set_ylim(ax.get_ylim()[0]-(y_lim[1]-y_lim[0])*box_widening, ax.get_ylim()[1]+(y_lim[1]-y_lim[0])*box_widening)
