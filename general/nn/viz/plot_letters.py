import numpy as np
from matplotlib.patches import Ellipse, Circle


def plot_N_on_ax(ax, center, height=1, width=None, **kwargs):
    if width is None:
        width = height

    segm1 = np.array([
        [center[0] - width / 2, center[0] - width / 2],  # xs coordinate
        [center[1] - height / 2, center[1] + height / 2]  # ys coordinate
    ])

    segm3 = np.array([
        [center[0] + width / 2, center[0] + width / 2],  # xs coordinate
        [center[1] - height / 2, center[1] + height / 2]  # ys coordinate
    ])

    segm2 = np.array([
        [center[0] - width / 2, center[0] + width / 2],  # xs coordinate
        [center[1] + height / 2, center[1] - height / 2]  # ys coordinate
    ])


    ax.plot(*segm1, **kwargs)
    ax.plot(*segm2, **kwargs)
    ax.plot(*segm3, **kwargs)

    return ax


def plot_O_on_ax(ax, center, height, width=None, **kwargs):
    if width is None:
        width = height

    ellipse = Ellipse(center, width, height, angle=0, fill=False, **kwargs)
    ax.add_patch(ellipse)

    return ax


def plot_NO_on_ax(ax, center, width, space_prop=.1, draw_circle=True, circle_fill="white", **kwargs):
    height = (1 - space_prop) * width / 2
    kwargs['color'] = kwargs.get('color', 'k')

    if draw_circle:
        radius = width / 2

        scale = 2 * width / (1.1 * np.sqrt(width ** 2 + width ** 2))
        width = width / scale
        height = height / scale

        ax.add_patch(Circle(center, radius, fill=True, color=circle_fill))
        ax.add_patch(Circle(center, radius, fill=False, **kwargs))

    letter_width = (1 - space_prop) * width / 2

    cursor = letter_width / 2

    plot_N_on_ax(ax, np.array([cursor + center[0] - width / 2, center[1]]), height=height, width=letter_width, **kwargs)
    cursor += letter_width

    cursor += space_prop * width

    plot_O_on_ax(ax, np.array([cursor + center[0] - width / 2, center[1]]), height=height, width=letter_width, **kwargs)

    return ax
