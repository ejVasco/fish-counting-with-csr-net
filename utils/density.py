# script for generating density maps

import numpy as np
from scipy.ndimage import gaussian_filter


def gen_density_map(image_shape, points, sigma =4):
    """
    Args
        auau
    Returns
        density_map:
    """

    H, W = image_shape
    density = np.zeros((H,W), dtype=np.float32)

    if len(points) ==0:
        return density # exit if no labeled points in json
    
    for x, y in points:
        x = int(round(x))
        y = int(round(y))
        
        if x<0 or x>=W or y<0 or y>=H:
            continue # skips if somehow there are points outside the image
        
        density[y,x] += 1.0 # set density at point locations
    
    density = gaussian_filter(density, sigma=sigma)
    
    return density