#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


"""

import numpy as np
from scipy.ndimage.filters import gaussian_filter


# --- Heatmap utilities ---
def get_heatmap_sci(x, y, sigma=None, framesize = [1000,1000] ):
    if sigma is None: 
        raise ValueError('Be sure that sigma value is proper!')
    
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=framesize, range = [[0,framesize[0]],[0,framesize[1]]])
    heatmap = gaussian_filter(heatmap.T, sigma=sigma,mode='constant',cval=0.0) # needed .T due to histogram2d's output settings. 

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap, extent


def et_heatmap(et_xy_in, framesize, sigma, hp_down_factor, get_full=False, get_down=True,
               nan_ratio=1.):
    # et_xy_in: [n_samples, 2], columns are x and y components
    # framesize = [frame_width,frame_height]
    
    if et_xy_in.ndim == 1:
        et_xy_in = et_xy_in[np.newaxis]
    
    # here we can eliminate video blocks which contain more that 50% problematic ET points.
    if np.isnan(np.sum(et_xy_in,1)).sum() >= nan_ratio*et_xy_in.shape[0]:
        if get_down:
            heatmap_down = np.zeros((np.array(framesize[::-1])/hp_down_factor).astype(int))
        if get_full:
            heatmap = np.zeros((np.array(framesize[::-1])).astype(int))
    else:
        if get_down:
            heatmap_down, _ = get_heatmap_sci(et_xy_in[:,0]/hp_down_factor,et_xy_in[:,1]/hp_down_factor,
                             sigma=sigma/float(hp_down_factor),framesize=(np.array(framesize)/hp_down_factor).astype(int))
        if get_full:
            heatmap, _ = get_heatmap_sci(et_xy_in[:,0],et_xy_in[:,1],sigma=sigma,framesize=framesize)

    if get_down and get_full:
        return heatmap_down, heatmap
    if get_full and not get_down:
        return heatmap
    else:
        return heatmap_down


