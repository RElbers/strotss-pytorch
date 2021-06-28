import math

import numpy as np
import torch
from numpy.random import randint


def _sample_points(features, x_coords, y_coords):
    initial_h = features[0].shape[-2]
    initial_w = features[0].shape[-1]

    image_samples = []
    for feature_map in features:
        current_h, current_w = feature_map.shape[-2:]
        factor_h = initial_h / current_h
        factor_w = initial_w / current_w

        # Rescale coordinates if feature-map is downscaled
        x_idc = x_coords // factor_w
        y_idc = y_coords // factor_h

        image_samples.append(feature_map[..., y_idc, x_idc])

    image_samples = torch.cat(image_samples, dim=1)
    image_samples = torch.transpose(image_samples, dim0=1, dim1=2)
    return image_samples


def sample_points(*features, n_samples=1024):
    assert len(features) > 0

    x_coords, y_coords = _random_coordinates(features, n_samples)

    if len(features) == 1:
        return _sample_points(features[0], x_coords, y_coords)
    return [_sample_points(fs, x_coords, y_coords) for fs in features]


def _random_coordinates(features, n_samples):
    h, w = features[0][0].shape[-2:]

    x_coords = randint(w, size=n_samples)
    y_coords = randint(h, size=n_samples)
    return x_coords, y_coords


def _grid_coordinates(features, n_samples):
    assert math.sqrt(n_samples).is_integer()

    h, w = features[0][0].shape[-2:]
    size = int(math.sqrt(n_samples))

    start_x = 0 if w <= size else randint(w - size)
    start_y = 0 if h <= size else randint(h - size)

    y_coords, x_coords = np.meshgrid(range(h)[start_y:start_y + size],
                                     range(w)[start_x:start_x + size])

    x_coords = x_coords.flatten()
    y_coords = y_coords.flatten()
    # shuffle(x_coords)
    # shuffle(y_coords)
    return x_coords, y_coords
