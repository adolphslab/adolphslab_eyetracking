#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Some stats and math funtions copied from various sources. 
Needs references.

"""

import numpy as np


def nanmasked_mean(arr_,axis):
    # works as np.nanmean(arr_,axis=1) but not raises: RuntimeWarning: Mean of empty slice
    # masked_where is slower. masked_invalid almost the same speed. 
    mdat = np.ma.masked_array(arr_,np.isnan(arr_))
    #mdat = np.ma.masked_invalid(arr_) # masks infs as well. 
    mdat_mean = np.mean(mdat,axis=axis)
    # mdat_mean = np.ma.masked_array(arr_,np.isnan(arr_)).mean(axis=axis)
    return mdat_mean.filled(np.nan)






