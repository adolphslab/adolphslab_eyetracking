#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Some stats and math funtions copied from various sources. 
Needs references.

"""

import numpy as np
from scipy.stats import norm, zscore



def list_flatten(list_in):
    list_out = []

    for eii in list_in:
        if isinstance(eii, list):
            for tii in eii:
                list_out.append(tii)
        else:
            list_out.append(eii)
    return list_out


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
    
    instead of cross_correlation(A, A, axis=1) or cross_correlation(A, B=None, axis=1) 
    use np.corrcoef(xarr,rowvar=True)
    
    or instead of cross_correlation(A, A, axis=0) use np.corrcoef(xarr,rowvar=False); 
    np.corrcoeff() is faster.
    
    '''
    n_row, n_col = A.shape
    A = zscore(A,axis)
    B = zscore(B,axis)
    if axis == 1:
        corr = np.dot(A, B.T)/float(n_col)
    elif axis ==0:    
        corr = np.dot(A.T, B)/float(n_row)
    return corr


#%% --- Effect size utilities ---
def cohen_d(x,y,rm_nan=False):
    # Computes Cohen's d for two arrays.
    if rm_nan:
        x = np.array(x, copy=True)
        y = np.array(y, copy=True)
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]

    nx, ny = len(x) , len(y)
    dof = nx + ny - 2.
    
    return (np.mean(x) - np.mean(y)) / np.sqrt( ((nx-1)*(np.std(x, ddof=1)**2.) + 
                                                 (ny-1)*(np.std(y, ddof=1)**2.) ) / dof)


from scipy.special import gamma
def hedge_g(x,y):
    nx, ny = len(x) , len(y)
    dof = float(nx + ny - 2.)
    if (nx+ny) <= 40: g_coeff = ( gamma(dof/2.) ) / ( np.sqrt(dof/2.)*gamma((dof-1.)/2.) )
    else: g_coeff = ( 1. - 3./(4.*(nx+ny) - 9.) )
    return g_coeff*(np.mean(x)-np.mean(y)) / np.sqrt(((nx-1)*(np.std(x,ddof=1)**2.)+(ny-1)*(np.std(y,ddof=1)**2.))/dof)


def cohen_d_mat(x2d, y2d, rm_extreme=False, n_std=4):
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
    assert x2d.shape[1] == y2d.shape[1], "Problem in computing Cohen's d"

    nx_cols = np.sum(~np.isnan(x2d),axis=0).astype(float)
    ny_cols = np.sum(~np.isnan(y2d),axis=0).astype(float)
    dof = nx_cols + ny_cols - 2.

    x2d_z, y2d_z = x2d.copy(), y2d.copy()
    x2d_z[x2d_z==1] = 1 - 1e-16 # Against an error in arctanh. 
    y2d_z[y2d_z==1] = 1 - 1e-16 # Against an error in arctanh. 

    x2d_z, y2d_z = np.arctanh(x2d_z), np.arctanh(y2d_z)

    # --- Cohen's d with the pooled standard deviation: similar to Hedge's g --- 
    return (np.nanmean(x2d_z,axis=0) - np.nanmean(y2d_z,axis=0)) / np.sqrt( ( (nx_cols-1)*(np.nanstd(x2d_z,axis=0,ddof=1)**2.) + 
                       (ny_cols-1)*(np.nanstd(y2d_z,axis=0,ddof=1)**2.) ) / dof)

    # --- Classical form introduced by Cohen. sqrt( (s1**2+s2**2)/2.) --- 
#    return (np.nanmean(x2d_z,axis=0) - np.nanmean(y2d_z,axis=0)) / np.sqrt( ( (np.nanstd(x2d_z,axis=0,ddof=1)**2.) + 
#                       (np.nanstd(y2d_z,axis=0,ddof=1)**2.) ) / 2.)


def hedges_g_mat_z(x2d,y2d):
    # Computes Hedges's g for each column separately.
    assert x2d.shape[1] == y2d.shape[1], "Problem in computing Hedges's g"
    
    hedges_vals = [] 
    for col_ii in range(x2d.shape[1]):
        x = x2d[:,col_ii].copy()
        y = y2d[:,col_ii].copy()
        
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]
        
        x[x==1] = 1 - 1e-16
        y[y==1] = 1 - 1e-16
        
        x, y = np.arctanh(x), np.arctanh(y)
        
        nx, ny = len(x) , len(y)
        dof = float(nx + ny - 2.)
        
        if (nx+ny) <= 40: g_coeff = ( gamma(dof/2.) ) / ( np.sqrt(dof/2.)*gamma((dof-1.)/2.) )
        else: g_coeff = ( 1. - 3./(4.*(nx+ny) - 9.) )
    
        hedges_vals.append( g_coeff*(np.mean(x)-np.mean(y)) / np.sqrt(((nx-1)*(np.std(x,ddof=1)**2.)+(ny-1)*(np.std(y,ddof=1)**2.))/dof))
    return np.asarray(hedges_vals)



def rm_outlier(arr_1,arr_2,n_std=4.,return_mask=False):
    # arr_1 and arr_2: 1D arrays. 
    arr_comb = np.r_[arr_1,arr_2]
    x_out = outlier_std(arr_comb,n_std=n_std)
    x_out_1, x_out_2, x_em = np.array_split(x_out,[len(arr_1),len(arr_1)+len(arr_2)])
    assert len(x_em)==0 and len(x_out_1)==len(arr_1) and len(x_out_2)==len(arr_2)
    
    if return_mask:
        return arr_1[~x_out_1], arr_2[~x_out_2], x_out_1, x_out_2
    
    return arr_1[~x_out_1], arr_2[~x_out_2]



def cohen_d_ci(x, y, confidence=0.95, n_boot=10000, n_sample=None,
               method='per', decimals=5, seed=None,  
               return_dist=False, alternative='two-sided', rm_nan=False, 
               rm_extreme=False, r2z_xform=False):
    
    # confidence=.68 for std.
    # x, y: 1d array.
    # confidence = 0.95
    # default percentile CI to make it consistent with p-value. 
    
    assert method in ['norm', 'normal', 'percentile', 'per', 'bca', 'cper']
    assert alternative in ['two-sided', 'one-sided']
    assert isinstance(confidence, float) and 0 < confidence < 1
    
    
    x = np.array(x)
    y = np.array(y)

    if r2z_xform:
        x[x==1] = 1.0 - 1e-16
        y[y==1] = 1.0 - 1e-16
        x = np.arctanh(x)
        y = np.arctanh(y)

    if rm_nan:
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]

    if rm_extreme:
        x, y, x_out_mask, y_out_mask = rm_outlier(x,y,n_std=3.,return_mask=True)
        if np.any(np.r_[x_out_mask,y_out_mask]): 
            print('removed x: %d, y: %d'%(sum(x_out_mask),sum(y_out_mask)))
    
    nx = len(x)
    assert x.ndim == 1 and nx > 1

    ny = len(y)
    assert y.ndim == 1 and ny > 1
    
    
    assert (nx >= 15 and ny >= 15), \
        'Not enough sample size for computation: nx: %d, ny: %d '%(nx, ny)

    if n_sample is not None:
        n_sample_x = n_sample
        n_sample_y = n_sample
    else:
        n_sample_x = nx
        n_sample_y = ny
        

    # Bootstrap process
    rng = np.random.default_rng(seed=seed)
    inds_x = rng.integers(nx, size=(n_sample_x, n_boot))
    inds_y = rng.integers(ny, size=(n_sample_y, n_boot))

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

    # Compute bootstrap p-value.
    p_val = bootstrap_pval(bootstat, reference, alternative)
    
    if return_dist:
        return reference.round(decimals), np.mean(bootstat).round(decimals), \
            [ ci[0].round(decimals), ci[1].round(decimals) ], p_val.round(decimals), bootstat

    return reference.round(decimals), np.mean(bootstat).round(decimals), \
        [ ci[0].round(decimals), ci[1].round(decimals) ], p_val.round(decimals)



from scipy.stats import rankdata
from scipy.stats import pearsonr, spearmanr

def bootstrap_pval(x_boots, x_estimate=None, alternative='two-sided'):
    """
    Compute p-value from a bootstrap distribution.
    
    The null hypothesis is that the mean of the distribution 
    from which x is sampled is zero.
    
    Parameters
    ----------
    x_boots : array_like
        The bootstrap sample.
    x_estimate : float
        The original estimate of x.
    alternative : str, optional
        Specifies the alternative hypothesis ('two-sided' or 'one-sided').
    
    Returns
    -------
    float
        The computed p-value.
    """
    
    x_boots = np.array(x_boots)
    assert x_boots.ndim == 1
    assert alternative in ['two-sided', 'one-sided']
    
    if x_estimate is None:
        x_estimate = np.mean(x_boots)
    
    # compute for one-sided
    if x_estimate == 0:
        p_val = 0.5
    else:
        p_val = min( np.sum(x_boots > 0)+1, np.sum(x_boots < 0)+1 ) / (len(x_boots)+1)
    
    # Return the appropriate p-value based on the 'alternative' parameter
    if alternative=='two-sided':
        p_val = 2 * p_val

    return min(p_val, 1.0)


def bootstrap_pval_oldv(x_boots, x_estimate=None, alternative='two-sided'):
    """
    
    Older version
    
    Compute p-value from a bootstrap distribution.
    
    The null hypothesis is that the mean of the distribution 
    from which x is sampled is zero.

    Parameters
    ----------
    x_boots : array_like
        The bootstrap sample.
    x_estimate : float
        The original estimate of x.
    alternative : str, optional
        Specifies the alternative hypothesis ('two-sided' or 'one-sided').
    
    Returns
    -------
    float
        The computed p-value.
    """

    x = np.array(x_boots)
    assert x.ndim == 1
    assert alternative in ['two-sided', 'one-sided']

    if x_estimate is None:
        x_estimate = np.mean(x_boots)
    
    if x_estimate > 0:
        p_1 = (np.sum(x<=0)+1.)/(x.size+1.)
        p_2 = (np.sum(x>0)+1.)/(x.size+1.)
    elif x_estimate < 0:
        p_1 = (np.sum(x<0)+1.)/(x.size+1.)
        p_2 = (np.sum(x>=0)+1.)/(x.size+1.)
    else: 
        p_1 = (np.sum(x<=0)+1.)/(x.size+1.)
        p_2 = (np.sum(x>=0)+1.)/(x.size+1.)

    # Return the appropriate p-value based on the 'alternative' parameter
    if alternative=='two-sided':
        return 2.*np.min([p_1,p_2])
    elif alternative=='one-sided':
        return np.min([p_1,p_2])


def rankzscore(x, axis):
    """
    Compute the z-score of the rank-transformed data along a specified axis.

    Parameters
    ----------
    x : array_like
        The data array to transform.
    axis : int
        The axis along which to compute the ranks and z-scores.

    Returns
    -------
    array_like
        The z-scores of the rank-transformed data.
    """
    return zscore(np.apply_along_axis(rankdata, axis, x))


def _bootstrap_pearson(x, y, boot_inds, axis=1):
    '''
    Function to bootstrap Pearson correlation

    if axis=1, compares rows of x[boot_inds] and y[boot_inds], 
    i.e., boots_inds has the shape of (n_samples, x.shape[0]), 
    
    if axis=0, compares columns of x[boot_inds] and y[boot_inds], 
    i.e., boots_inds has the shape of (x.shape[0], n_samples)
    '''
    return (zscore(x[boot_inds], axis=axis) * zscore(y[boot_inds], axis=axis)).mean(axis=axis)

def _bootstrap_spearman(x, y, boot_inds, axis=1):
    # Function to bootstrap Spearman correlation
    return (rankzscore(x[boot_inds], axis) * rankzscore(y[boot_inds], axis)).mean(axis=axis)

def _bootstrap_mean_diff(x, y, boot_inds, boot_inds2, axis=1):
    # Function to bootstrap mean difference
    return np.mean(x[boot_inds], axis=axis) - np.mean(y[boot_inds2], axis=axis)


stat_funcs = {'pearson': lambda a, b: pearsonr(a, b)[0],
              'spearman': lambda a, b: spearmanr(a, b)[0],
              'mean_diff': lambda a, b: np.mean(a) - np.mean(b)}


def bootstrap_ci(x, y, func='pearson', confidence=0.95, n_boots=10000, n_samples=None,
                 seed=None, alternative='two-sided', return_dist=False, decimals=5):
    """
    Calculate the bootstrap confidence interval for a given statistic.

    Parameters
    ----------
    x, y : array_like
        Input data arrays.
    func : str, optional
        The statistic to be computed. Must be 'pearson', 'spearman', or 'mean_diff'.
    confidence : float, optional
        The confidence level for the interval.
    n_boots : int, optional
        The number of bootstrap samples to draw.
    n_samples : int or None, optional
        The number of samples to draw in each bootstrap iteration, makes it like jackknife.
    seed : int or None, optional
        Seed for the random number generator.
    alternative : str, optional
        Specifies the alternative hypothesis ('two-sided' or 'one-sided').
    return_dist : bool, optional
        If True, returns the bootstrap distribution.

    Returns
    -------
    tuple
        Tuple containing the original statistic, mean of bootstrap distribution, 
        confidence interval, p-value, and optionally the bootstrap distribution.
    """
    
    # Validation of inputs
    assert func in [ 'pearson', 'spearman', 'mean_diff'], f'func={func} is undefined!'
    assert alternative in ['two-sided', 'one-sided']
    assert isinstance(confidence, float) and 0 < confidence < 1

    x = np.array(x)
    nx = len(x)
    assert x.ndim == 1 and nx > 1

    y = np.array(y)
    ny = len(y)
    assert y.ndim == 1 and ny > 1

    # if not, we can use: n_sample = min(nx, ny); but be careful!
    assert nx == ny
    
    if n_samples is None:
        n_samples = nx
        
    alpha = 1 - confidence
    alphas = np.array([alpha/2, 1-alpha/2])

    # Compute the original statistic based on the specified function.
    org_stat = stat_funcs[func](x, y)
    
    # Bootstrap process
    rng = np.random.default_rng(seed=seed)
    boot_inds = rng.integers(x.shape[0], size=(n_boots, n_samples))
    if func=='mean_diff':
        boot_inds2 = rng.integers(y.shape[0], size=(n_boots, n_samples))
    
    # Compute the bootstrap statistic based on the specified function.
    if func == 'pearson':
        boots_dist = _bootstrap_pearson(x, y, boot_inds)
    elif func == 'spearman':
        boots_dist = _bootstrap_spearman(x, y, boot_inds)
    elif func == 'mean_diff':
        boots_dist = _bootstrap_mean_diff(x, y, boot_inds, boot_inds2)
    
    # Confidence interval and p-values calculation
    ci = np.percentile(boots_dist, alphas*100)
    p_val = bootstrap_pval(boots_dist, org_stat, alternative)

    if return_dist:
        return org_stat, np.mean(boots_dist), ci, p_val, boots_dist
    
    if decimals is None:
        return org_stat, np.mean(boots_dist), ci, p_val
    else:
        return org_stat.round(decimals), np.mean(boots_dist).round(decimals),\
            ci.round(decimals), p_val.round(decimals)
