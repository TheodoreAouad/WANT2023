import numpy as np
from PIL import Image, ImageDraw


def rand_shape_2d(shape, rng_float=lambda shape: np.random.rand(shape[0], shape[1])):
    try:
        return rng_float(shape)
    except TypeError:
        return rng_float(*shape)


def straight_rect(width, height):
    return np.array([(0, 0), (width, 0), (width, height), (0, height), (0, 0)])


def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


def invert_proba(ar, p_invert: float, rng_float) -> np.ndarray:
    if not isinstance(ar, list):
        if rng_float() < p_invert:
            return 1 - ar
        return ar

    if rng_float() < p_invert:
        res = []
        for a in ar:
            res.append(1 - a)
        return res
    return ar


def get_rect_vertices(x, y, width, height, angle):
    rect = straight_rect(width, height)
    theta = (np.pi / 180.0) * angle
    R = rotation_matrix(theta)
    offset = np.array([x, y])
    transformed_rect = np.dot(rect, R) + offset
    return transformed_rect


def draw_poly(draw, poly, fill_value=1):
    draw.polygon([tuple(p) for p in poly], fill=fill_value)


def draw_ellipse(draw, center, radius, fill_value=1):
    bbox = (center[0] - radius[0], center[1] - radius[1], center[0] + radius[0], center[1] + radius[1])
    draw.ellipse(bbox, fill=fill_value)


def get_rect(width, height, angle):
    max_dim = max(height, width)

    if max_dim % 2 == 0:
        max_dim += 1

    theta = (np.pi / 180.0) * angle

    y = (max_dim - (height * np.cos(theta))) / 2
    x = (max_dim - (height * np.sin(theta))) / 2


    ar = np.zeros((max_dim, max_dim))
    # ar = np.zeros((int(shape_x) + 1, int(shape_y) + 1))
    img = Image.fromarray(ar)

    draw = ImageDraw.Draw(img)
    draw_poly(draw, get_rect_vertices(x, y, width, height, angle), fill_value=1)

    ar = np.asarray(img) + 0

    return ar


