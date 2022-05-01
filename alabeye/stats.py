#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Some stats and math funtions copied from various sources. 
Needs references.

"""

import numpy as np
from scipy.stats import norm, zscore


def nanmasked_mean(arr_,axis):
    # works as np.nanmean(arr_,axis=1) but not raises: RuntimeWarning: Mean of empty slice
    # masked_where is slower. masked_invalid almost the same speed. 
    mdat = np.ma.masked_array(arr_,np.isnan(arr_))
    #mdat = np.ma.masked_invalid(arr_) # masks infs as well. 
    mdat_mean = np.mean(mdat,axis=axis)
    # mdat_mean = np.ma.masked_array(arr_,np.isnan(arr_)).mean(axis=axis)
    return mdat_mean.filled(np.nan)



def outlier_std(x,y=None,n_std=4.):
    x = np.asarray(x)
    assert x.ndim ==1
    
    x_mean = x.mean()
    x_std = x.std()
    x_out = np.logical_or( x >= (x_mean + n_std*x_std), x <= (x_mean - n_std*x_std) )

    if not y is None:
        y = np.asarray(y)
        y_mean = y.mean()
        y_std = y.std()
        y_out = np.logical_or( y >= (y_mean + n_std*y_std), y <= (y_mean - n_std*y_std) )
        return x_out, y_out
    
    return x_out



def cross_correlation(A, B, axis=1): # default rows
    '''Compute correlation for each row/column of A against every row/column of B.
    Not in ideal speed but has a broad use. 
    # see also: stackoverflow.com/questions/19401078/efficient-columnwise-correlation-coefficient-calculation-with-numpy
    
    instead of cross_correlation(A, A, axis=1) or cross_correlation(A, B=None, axis=1) use np.corrcoef(xarr,rowvar=True)
    or instead of cross_correlation(A, A, axis=0) use np.corrcoef(xarr,rowvar=False); np.corrcoeff() is faster.
    
    '''
    n_row, n_col = A.shape
    A = zscore(A,axis)
    B = zscore(B,axis)
    if axis == 1:
        corr = np.dot(A, B.T)/float(n_col)
    elif axis ==0:    
        corr = np.dot(A.T, B)/float(n_row)
    return corr


#%%

# --- Effect size utilities ---
def cohen_d(x,y,rm_nan=False):
    # Computes Cohen's d for two arrays.
    if rm_nan:
        x = np.array(x)
        y = np.array(y)
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]

    nx, ny = len(x) , len(y)
    dof = nx + ny - 2.
    
    return (np.mean(x) - np.mean(y)) / np.sqrt( ( (nx-1)*(np.std(x, ddof=1)**2.) + (ny-1)*(np.std(y, ddof=1)**2.) ) / dof)



from scipy.special import gamma
def hedge_g(x,y):
    nx, ny = len(x) , len(y)
    dof = float(nx + ny - 2.)
    if (nx+ny) <= 40: g_coeff = ( gamma(dof/2.) ) / ( np.sqrt(dof/2.)*gamma((dof-1.)/2.) )
    else: g_coeff = ( 1. - 3./(4.*(nx+ny) - 9.) )
    return g_coeff*(np.mean(x)-np.mean(y)) / np.sqrt(((nx-1)*(np.std(x,ddof=1)**2.)+(ny-1)*(np.std(y,ddof=1)**2.))/dof)



def cohen_d_mat(x2d,y2d,rm_extreme=False):
    # Computes Cohen's d for each column separately.
    if x2d.shape[1] != y2d.shape[1]: 
        raise SystemExit("Array dims mismatch in computing Cohen's d")

    if rm_extreme:
        x2d = np.array(x2d,copy=True)
        y2d = np.array(y2d,copy=True)
        
        col_mean = np.nanmean(np.vstack((x2d,y2d)), 0)
        col_std = np.nanstd(np.vstack((x2d,y2d)), 0)
        
        x2d_temp = np.nan_to_num(x2d)
        y2d_temp = np.nan_to_num(y2d)

        n_std=4.
        x_out = np.logical_or( x2d_temp >= (col_mean + n_std*col_std), x2d_temp <= (col_mean - n_std*col_std) )
        y_out = np.logical_or( y2d_temp >= (col_mean + n_std*col_std), y2d_temp <= (col_mean - n_std*col_std) )
        
        x2d[x_out] = np.NaN
        y2d[y_out] = np.NaN


    nx_cols = np.sum(~np.isnan(x2d),axis=0).astype(float)
    ny_cols = np.sum(~np.isnan(y2d),axis=0).astype(float)
    dof = nx_cols + ny_cols - 2.

    # --- Cohen's d with the pooled standard deviation: similar to Hedge's g --- 
    return (np.nanmean(x2d,axis=0) - np.nanmean(y2d,axis=0)) / np.sqrt( ( (nx_cols-1)*(np.nanstd(x2d,axis=0,ddof=1)**2.) + 
                       (ny_cols-1)*(np.nanstd(y2d,axis=0,ddof=1)**2.) ) / dof)

    # --- Classical form introduced by Cohen. sqrt( (s1**2+s2**2)/2.) --- 
    # return (np.nanmean(x2d,axis=0) - np.nanmean(y2d,axis=0)) / np.sqrt( ( (np.nanstd(x2d,axis=0,ddof=1)**2.) + 
                      # (np.nanstd(y2d,axis=0,ddof=1)**2.) ) / 2.)


def cohen_d_mat_z(x2d,y2d):
    # Computes Cohen's d for each column separately after Fisher-z transformation.
    if x2d.shape[1] != y2d.shape[1]: raise SystemExit("Problem in computing Cohen's d")

    nx_cols = np.sum(~np.isnan(x2d),axis=0).astype(float)
    ny_cols = np.sum(~np.isnan(y2d),axis=0).astype(float)
    dof = nx_cols + ny_cols - 2.

    x2d_z, y2d_z = x2d.copy(), y2d.copy()
    x2d_z[x2d_z==1] = 0.99999 # Against an error in arctanh. 
    y2d_z[y2d_z==1] = 0.99999 # Against an error in arctanh. 

    x2d_z, y2d_z = np.arctanh(x2d_z), np.arctanh(y2d_z)

    # --- Cohen's d with the pooled standard deviation: similar to Hedge's g --- 
    return (np.nanmean(x2d_z,axis=0) - np.nanmean(y2d_z,axis=0)) / np.sqrt( ( (nx_cols-1)*(np.nanstd(x2d_z,axis=0,ddof=1)**2.) + 
                       (ny_cols-1)*(np.nanstd(y2d_z,axis=0,ddof=1)**2.) ) / dof)

    # --- Classical form introduced by Cohen. sqrt( (s1**2+s2**2)/2.) --- 
#    return (np.nanmean(x2d_z,axis=0) - np.nanmean(y2d_z,axis=0)) / np.sqrt( ( (np.nanstd(x2d_z,axis=0,ddof=1)**2.) + 
#                       (np.nanstd(y2d_z,axis=0,ddof=1)**2.) ) / 2.)


def hedges_g_mat_z(x2d,y2d):
    # Computes Hedges's g for each column separately.
    if x2d.shape[1] != y2d.shape[1]: raise SystemExit("Problem in computing Hedges's g")
    hedges_vals = [] 
    for col_ii in range(x2d.shape[1]):
        x = x2d[:,col_ii].copy()
        y = y2d[:,col_ii].copy()
        
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]
        
        x[x==1] = 0.99999
        y[y==1] = 0.99999
        
        x, y = np.arctanh(x), np.arctanh(y)
        
        nx, ny = len(x) , len(y)
        dof = float(nx + ny - 2.)
        
        if (nx+ny) <= 40: g_coeff = ( gamma(dof/2.) ) / ( np.sqrt(dof/2.)*gamma((dof-1.)/2.) )
        else: g_coeff = ( 1. - 3./(4.*(nx+ny) - 9.) )
    
        hedges_vals.append( g_coeff*(np.mean(x)-np.mean(y)) / np.sqrt(((nx-1)*(np.std(x,ddof=1)**2.)+(ny-1)*(np.std(y,ddof=1)**2.))/dof))
    return np.asarray(hedges_vals)



def rm_outlier(arr_1,arr_2,n_std=4,return_mask=False):
    # arr_1 and arr_2: 1D arrays. 
    arr_comb = np.r_[arr_1,arr_2]
    x_out = outlier_std(arr_comb,n_std=n_std)
    x_out_1, x_out_2, x_em = np.array_split(x_out,[len(arr_1),len(arr_1)+len(arr_2)])
    assert len(x_em)==0 and len(x_out_1)==len(arr_1) and len(x_out_2)==len(arr_2)
    
    if return_mask:
        return arr_1[~x_out_1], arr_2[~x_out_2], x_out_1, x_out_2
    
    return arr_1[~x_out_1], arr_2[~x_out_2]





def cohen_d_ci(x,y, confidence=.95, n_boot=20000, method='per', decimals=3, return_dist=False, 
               tail='two-sided', rm_nan=False, rm_extreme=False):
    # confidence=.68 for std.
    
    # x, y: 1d array.
    # confidence = 0.95
    # default percentile CI to make it consistent with p-value. 
    
    x = np.array(x, copy=True)
    y = np.array(y, copy=True)

    if rm_nan:
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]

    
    if rm_extreme:
        x, y, x_out_mask, y_out_mask = rm_outlier(x,y,n_std=4,return_mask=True)
        if np.any(np.r_[x_out_mask,y_out_mask]): 
            print('removed x: %d, y: %d'%(sum(x_out_mask),sum(y_out_mask)))
    
    n_x = x.size
    assert x.ndim == 1
    assert n_x > 1

    n_y = y.size
    assert y.ndim == 1
    assert n_y > 1

    assert (n_x >= 15 and n_y >= 15), 'Not enough sample size for computation: n_x: %d, n_y: %d '%(n_x,n_y)

    assert isinstance(confidence, float)
    assert 0 < confidence < 1
    if not method in ['normal', 'percentile', 'per', 'cper', 'bca' ]:
        raise ValueError('Undefined method in cohen_d_CI!')


    inds_x = np.random.randint(n_x, size=(n_x,n_boot))
    inds_y = np.random.randint(n_y, size=(n_y,n_boot))
    # inds_x = np.random.choice(n_x, size=(n_x,n_boot),replace=True)          
    # inds_y = np.random.choice(n_y, size=(n_y,n_boot),replace=True)          
    
    x_sample = x[inds_x]
    y_sample = y[inds_y]
    
    reference = cohen_d(x,y)
    bootstat = cohen_d_mat(x_sample,y_sample)
    
    # Confidence intervals
    alpha = 1 - confidence
    dist_sorted = np.sort(bootstat)
    
    
    if method in ['norm', 'normal']:
        # Normal approximation
        za = norm.ppf(alpha / 2)
        se = np.std(bootstat, ddof=1)

        bias = np.mean(bootstat - reference)
        ll = reference - bias + se * za
        ul = reference - bias - se * za
        ci = [ll, ul]
    elif method in ['percentile', 'per']:
        # Uncorrected percentile
        pct_ll = int(n_boot * (alpha / 2))
        pct_ul = int(n_boot * (1 - alpha / 2))
        ci = [dist_sorted[pct_ll], dist_sorted[pct_ul]]
    elif method in ['bca', 'cper']:
        # Corrected percentile bootstrap
        # Compute bias-correction constant z0
        z_0 = norm.ppf(np.mean(bootstat < reference) +
                       np.mean(bootstat == reference) / 2)
        z_alpha = norm.ppf(alpha / 2)
        pct_ul = 100 * norm.cdf(2 * z_0 - z_alpha)
        pct_ll = 100 * norm.cdf(2 * z_0 + z_alpha)
        ll = np.percentile(bootstat, pct_ll)
        ul = np.percentile(bootstat, pct_ul)
        ci = [ll, ul]


    ci = np.round(ci, decimals)
    # Compute bootstrap p-value.
    pval = (np.sum(bootstat<=0.)+1.)/float(len(bootstat)+1.) if np.mean(bootstat)>=0 else (np.sum(bootstat>=0.)+1.)/float(len(bootstat)+1.)
    if tail == 'two-sided':
        pval = pval*2.

    if return_dist:
        return reference, np.mean(bootstat), ci[0], ci[1], pval, bootstat
    else:
        return reference, np.mean(bootstat), ci[0], ci[1], pval

