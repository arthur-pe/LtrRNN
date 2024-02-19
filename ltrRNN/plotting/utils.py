import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager
import matplotlib
import os
import sys

default_cmap = matplotlib.colormaps['hsv']
axes_line_width = 1.5
default_font_size = 14


def plot_with_gradient_3d(ax, xs, ys, zs, gradient, cmap,
                          dashed=False, number_dashes=50, dash_density=0.7,
                          linewidth=1.0, alpha=1.0, zorder=3, set_lim=True):

    points = np.array([xs, ys, zs]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    number_segments = len(segments)

    if dashed:
        temp = []
        temp_cols = []
        for i in range(number_dashes):
            temp.append(segments[int(i*number_segments/number_dashes):int((i+dash_density)*number_segments/number_dashes)])
            temp_cols.append(gradient[int(i*number_segments/number_dashes):int((i+dash_density)*number_segments/number_dashes)])

        segments = np.concatenate(temp)
        gradient = np.concatenate(temp_cols)

    capstyle = 'round' if dashed or alpha==1 else 'butt'
    lc = Line3DCollection(segments, cmap=cmap, alpha=alpha, zorder=zorder, capstyle=capstyle)
    lc.set_array(gradient)
    lc.set_clim(0,1)
    lc.set_linewidth(linewidth)
    line = ax.add_collection(lc)

    if set_lim:
        ax.set_xlim(min((np.min(xs), ax.get_xlim()[0])), max((np.max(xs), ax.get_xlim()[1])))
        ax.set_ylim(min((np.min(ys), ax.get_ylim()[0])), max((np.max(ys), ax.get_ylim()[1])))
        ax.set_zlim(min((np.min(zs), ax.get_zlim()[0])), max((np.max(zs), ax.get_zlim()[1])))


def plot_with_gradient_2d(ax, xs, ys, gradient, cmap,
                          dashed=False, number_dashes=50, dash_density=0.7,
                          linewidth=1.0, alpha=1.0, zorder=3, set_lim=True):

    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    number_segments = len(segments)

    if dashed:
        temp = []
        temp_cols = []
        for i in range(number_dashes):
            temp.append(segments[int(i*number_segments/number_dashes):int((i+dash_density)*number_segments/number_dashes)])
            temp_cols.append(gradient[int(i*number_segments/number_dashes):int((i+dash_density)*number_segments/number_dashes)])
        segments = np.concatenate(temp)
        gradient = np.concatenate(temp_cols)

    capstyle = 'round' if dashed or alpha==1 else 'butt'
    lc = LineCollection(segments, cmap=cmap, alpha=alpha, zorder=zorder, capstyle=capstyle)
    lc.set_array(gradient)
    lc.set_clim(0,1)
    lc.set_linewidth(linewidth)
    line = ax.add_collection(lc)

    if set_lim:
        ax.set_xlim(min((np.min(xs), ax.get_xlim()[0])), max((np.max(xs), ax.get_xlim()[1])))
        ax.set_ylim(min((np.min(ys), ax.get_ylim()[0])), max((np.max(ys), ax.get_ylim()[1])))


def get_cmap_interpolated(*args):

    colors = []
    for i in range(len(args)-1):
        colors.append(np.stack([np.linspace(args[i][0],args[i+1][0],1001),
                                np.linspace(args[i][1],args[i+1][1],1001),
                                np.linspace(args[i][2],args[i+1][2],1001),
                                np.linspace(args[i][3] if len(args[i]) == 4 else 1,
                                            args[i+1][3] if len(args[i+1]) == 4 else 1, 1001)], axis=-1))
    colors = np.concatenate(colors, axis=0)
    cmap = LinearSegmentedColormap.from_list('interpolated_cmap', colors)

    return cmap


def get_cmap_black(color):

    return get_cmap_interpolated((0,0,0,1), color)


def get_cmap_white(color):

    return get_cmap_interpolated((1,1,1,1), color)


def set_pannels_3d(ax, x=False, y=False, z=True, grey=0.95):

    c_white = np.ones(3)
    c_grey = grey*c_white

    ax.xaxis.set_pane_color(c_grey if x else c_white)
    ax.yaxis.set_pane_color(c_grey if y else c_white)
    ax.zaxis.set_pane_color(c_grey if z else c_white)

    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)


def decorator_set_pannels_3d(func, x=False, y=False, z=True, grey=0.95):

    def decorator(*args, **kwargs):

        func(*args, **kwargs)

        ax = args[0]

        set_pannels_3d(ax, x, y, z, grey)

    return decorator


def get_ax_3d(figsize=(4,4), constrained_layout=True, dpi=100):

    fig = plt.figure(figsize=figsize, constrained_layout=constrained_layout, dpi=dpi)
    ax = fig.add_subplot(projection='3d')

    return ax


def get_ax_2d(figsize=(4, 4), constrained_layout=True, dpi=100):
    fig = plt.figure(figsize=figsize, constrained_layout=constrained_layout, dpi=dpi)
    ax = fig.add_subplot()

    return ax

def get_ax_gridspec(rows, columns, size_ax=4, axs_3d=(), axs_ignored=()):

    fig = plt.figure(figsize=(columns*size_ax,rows*size_ax), constrained_layout=True, dpi=80)
    gs = fig.add_gridspec(ncols=columns,nrows=rows)

    axs = [[fig.add_subplot(gs[i,j], projection=('3d' if [i,j] in axs_3d else None)) for i in range(rows) if [i,j] not in axs_ignored] for j in range(columns)]

    return axs, gs, fig


def remove_ticks(ax):
    ax.set_xticks([], []), ax.set_yticks([], [])
    if ax.name == "3d": ax.set_zticks([], [])


def set_font(font_name='HelveticaNeue', font_size=default_font_size, font_color=(0,0,0), unicode_minus=True):

    font_path = [sys.path[1]]
    font_files = font_manager.findSystemFonts(fontpaths=font_path)
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)

    matplotlib.rcParams['font.family'] = font_name
    matplotlib.rcParams['font.size'] = font_size
    matplotlib.rcParams["axes.unicode_minus"] = unicode_minus

    matplotlib.rcParams['text.color'] = font_color
    matplotlib.rcParams['axes.labelcolor'] = font_color
    matplotlib.rcParams['xtick.color'] = font_color
    matplotlib.rcParams['ytick.color'] = font_color


def set_centered_axes(ax, zero_centered=True):

    if not zero_centered:
        y_max = np.max(np.abs(np.array(ax.get_ylim())))
        ax.set_ylim(-y_max, y_max)
        x_max = np.max(np.abs(np.array(ax.get_xlim())))
        ax.set_xlim(-x_max, x_max)

    if zero_centered:
        ax.spines['left'].set_position(('data',0))
        ax.spines['bottom'].set_position(('data',0))
    else:
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')

    ax.spines['left'].set_linewidth(axes_line_width)
    ax.spines['bottom'].set_linewidth(axes_line_width)
    ax.spines['left'].set_capstyle('round')
    ax.spines['bottom'].set_capstyle('round')

    ax.tick_params(width=axes_line_width)

    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.spines['left'].set_color((0, 0, 0, 0.2))
    ax.spines['bottom'].set_color((0, 0, 0, 0.2))
    ax.spines['left'].set_zorder(1)
    ax.spines['bottom'].set_zorder(1)

    # Show ticks in the left and lower axes only
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.xaxis.set_zorder(1)
    ax.yaxis.set_zorder(1)

    """ax.get_xaxis().get_offset_text().set_position((-x_max, y_max/50))
    ax.get_yaxis().get_offset_text().set_position((x_max/50, y_max))"""


def set_bottom_axis(ax, color=(0, 0, 0)):
    for side in ax.spines.keys():  # 'top', 'bottom', 'left', 'right'
        ax.spines[side].set_linewidth(axes_line_width)
        #ax.spines[side].set_color((0.2,0.2,0.2))
        ax.spines[side].set_capstyle('round')

    ax.spines['bottom'].set_color(color)
    ax.spines['right'].set_color((1, 1, 1, 0))
    ax.spines['top'].set_color((1, 1, 1, 0))
    ax.spines['left'].set_color((1, 1, 1, 0))
    ax.tick_params(width=axes_line_width, color=color)


def set_axes_linewidth(ax, color=None, linewidth=None):
    for side in ax.spines.keys():  # 'top', 'bottom', 'left', 'right'
        ax.spines[side].set_linewidth(linewidth if linewidth is not None else axes_line_width)
        ax.spines[side].set_color(color if color is not None else (0.2, 0.2, 0.2))
        ax.spines[side].set_capstyle('round')

    ax.xaxis.set_tick_params(width=linewidth)
    ax.yaxis.set_tick_params(width=linewidth)


def remove_axes(ax):
    for side in ax.spines.keys():
        ax.spines[side].set_color((1, 1, 1, 0))


def set_set_equal_lim(ax):

    x_max = np.max(np.abs(np.array(ax.get_xlim())))
    y_max = np.max(np.abs(np.array(ax.get_ylim())))

    max_max = max(x_max, y_max)

    ax.set_xlim(-max_max, max_max)
    ax.set_ylim(-max_max, max_max)


def set_centered_axes_3d(ax, c=(0.2,0.2,0.2), linewidth=1.5):

    ax.axis('off')

    ax_lims = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])

    ax.plot(ax_lims[0,:], [ax_lims[1,0],ax_lims[1,0]], [ax_lims[2,0],ax_lims[2,0]], color=c, linewidth=linewidth)
    ax.plot([ax_lims[0,0],ax_lims[0,0]], ax_lims[1,:], [ax_lims[2,0],ax_lims[2,0]], color=c, linewidth=linewidth)
    ax.plot([ax_lims[0,0], ax_lims[0,0]],[ax_lims[1,0],ax_lims[1,0]], ax_lims[2,:], color=c, linewidth=linewidth)

    ax.view_init(23, 45, roll=0)
