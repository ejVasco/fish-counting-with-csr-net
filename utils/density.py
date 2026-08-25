# utils/density.py
# script for generating density maps
# from image and json annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def gen_density_map(image_shape, points, sigma=4):
    """
    Generates a density map from annotated points

    Args
        image_shape (tuple of int): shape of output density map in height and width, which should match the size of the image annotated
        points (list of tuples): list of (x,y) coords representing locations of objects (fish in this project)
        sigma (float): standard deviation for guassian filter to make points into a smooth density map bump. default: 4
    Returns
        numpy.ndarray: 2d array representing the density map
    """

    # create 2d array of 0s for the density values
    H, W = image_shape
    density = np.zeros(
        (H, W), dtype=np.float32
    )  # specify density values will be 32b floats

    if len(points) == 0:
        return density  # exit if no labeled points in json

    for x, y in points:
        # rounds points from annotations to nearest int
        x = int(round(x))
        y = int(round(y))

        if x < 0 or x >= W or y < 0 or y >= H:
            continue  # skips if somehow labeled points are outside the image

        density[y, x] += 1.0  # set density at point locations

    density = gaussian_filter(density, sigma=sigma)  # smooths out the density map

    return density
