#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Resample time bins for computing correlation statistics.
Used for Figure 2 (C,F,I)

"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm


from alabeye.etdata import makedir
from alabeye.etutils import load_gazedata_split

from statsmodels.sandbox.stats.multicomp import multipletests
from scipy.stats import pearsonr, spearmanr 


tbin = 600 # sec [ 10, 20, 30, 60, 120, 300, 600]
n_samples = 10000

#%% Main directory for experiment data
root_dir = '/home/umit/Documents/Research_ET/AutismVids/avp_data'

output_dir = os.path.join(root_dir,'Results_v1','splitparts_data')
# set up the output directory
makedir(output_dir,sysexit=False)

# load prepared gaze features data for sampling video epochs and subjects
data_dir = os.path.join(root_dir,'Results_v1','Cohend_pool')

asd_subjs, td_subjs, asd_feats, td_feats, featnames, sample_inds = \
    load_gazedata_split(data_dir, tbin, n_samples=n_samples, get_pair_samples=True)

#%%
feat0_asd_split = asd_feats[:,:,featnames.index('Onscreen')]
feat0_td_split = td_feats[:,:,featnames.index('Onscreen')]

feat1_asd_split = asd_feats[:,:,featnames.index('Face')]
feat1_td_split = td_feats[:,:,featnames.index('Face')]

feat2_asd_split = asd_feats[:,:,featnames.index('Eye')]
feat2_td_split = td_feats[:,:,featnames.index('Eye')]

feat3_asd_split = asd_feats[:,:,featnames.index('Heatcorr')]
feat3_td_split = td_feats[:,:,featnames.index('Heatcorr')]

assert not np.all(feat3_asd_split==0,1).any()
assert not np.all(feat3_td_split==0,1).any()


#%%
feat0_corrs_asd, feat0_corrs_td = np.zeros(n_samples), np.zeros(n_samples)
feat1_corrs_asd, feat1_corrs_td = np.zeros(n_samples), np.zeros(n_samples)
feat2_corrs_asd, feat2_corrs_td = np.zeros(n_samples), np.zeros(n_samples)
feat3_corrs_asd, feat3_corrs_td = np.zeros(n_samples), np.zeros(n_samples)

feat0_corrs_asd_s, feat0_corrs_td_s = np.zeros(n_samples), np.zeros(n_samples)
feat1_corrs_asd_s, feat1_corrs_td_s = np.zeros(n_samples), np.zeros(n_samples)
feat2_corrs_asd_s, feat2_corrs_td_s = np.zeros(n_samples), np.zeros(n_samples)
feat3_corrs_asd_s, feat3_corrs_td_s = np.zeros(n_samples), np.zeros(n_samples)

for pii in tqdm(range(n_samples)):

    p1, p2 = sample_inds[pii] 
    p12, p22 = sample_inds[pii] # in general use this might be sample_inds2 (see load_gazedata_split)

    asd_ii = np.random.choice(feat1_asd_split.shape[1],feat1_asd_split.shape[1],replace=True)
    # we select the same number TDs as ASDs for correlation resampling.
    td_ii = np.random.choice(feat1_td_split.shape[1],feat1_asd_split.shape[1],replace=True)
    
    feat0_corrs_asd[pii] = pearsonr(feat0_asd_split[p1,asd_ii],feat0_asd_split[p2,asd_ii])[0]
    feat0_corrs_td[pii] = pearsonr(feat0_td_split[p1,td_ii],feat0_td_split[p2,td_ii])[0] 
    
    feat1_corrs_asd[pii] = pearsonr(feat1_asd_split[p1,asd_ii],feat1_asd_split[p2,asd_ii])[0]
    feat1_corrs_td[pii] = pearsonr(feat1_td_split[p1,td_ii],feat1_td_split[p2,td_ii])[0] 
    
    feat2_corrs_asd[pii] = pearsonr(feat2_asd_split[p12,asd_ii],feat2_asd_split[p22,asd_ii])[0]
    feat2_corrs_td[pii] = pearsonr(feat2_td_split[p12,td_ii],feat2_td_split[p22,td_ii])[0] 
    
    feat3_corrs_asd[pii] = pearsonr(feat3_asd_split[p1,asd_ii],feat3_asd_split[p2,asd_ii])[0]
    feat3_corrs_td[pii] = pearsonr(feat3_td_split[p1,td_ii],feat3_td_split[p2,td_ii])[0] 

    # ----- spearmanr -----
    feat0_corrs_asd_s[pii] = spearmanr(feat0_asd_split[p1,asd_ii],feat0_asd_split[p2,asd_ii])[0]
    feat0_corrs_td_s[pii] = spearmanr(feat0_td_split[p1,td_ii],feat0_td_split[p2,td_ii])[0] 

    feat1_corrs_asd_s[pii] = spearmanr(feat1_asd_split[p1,asd_ii],feat1_asd_split[p2,asd_ii])[0]
    feat1_corrs_td_s[pii] = spearmanr(feat1_td_split[p1,td_ii],feat1_td_split[p2,td_ii])[0] 
    
    feat2_corrs_asd_s[pii] = spearmanr(feat2_asd_split[p12,asd_ii],feat2_asd_split[p22,asd_ii])[0]
    feat2_corrs_td_s[pii] = spearmanr(feat2_td_split[p12,td_ii],feat2_td_split[p22,td_ii])[0] 
    
    feat3_corrs_asd_s[pii] = spearmanr(feat3_asd_split[p1,asd_ii],feat3_asd_split[p2,asd_ii])[0]
    feat3_corrs_td_s[pii] = spearmanr(feat3_td_split[p1,td_ii],feat3_td_split[p2,td_ii])[0] 


#%%
feat0_corrs_diff_s = feat0_corrs_td_s-feat0_corrs_asd_s
feat1_corrs_diff_s = feat1_corrs_td_s-feat1_corrs_asd_s
feat2_corrs_diff_s = feat2_corrs_td_s-feat2_corrs_asd_s
feat3_corrs_diff_s = feat3_corrs_td_s-feat3_corrs_asd_s

#%%

# --- Save outputs for plotting ---
np.save(os.path.join(output_dir,f'feat0_corrs_asd_{tbin}_s'), feat0_corrs_asd_s)
np.save(os.path.join(output_dir,f'feat0_corrs_td_{tbin}_s'), feat0_corrs_td_s)

np.save(os.path.join(output_dir,f'feat1_corrs_asd_{tbin}_s'), feat1_corrs_asd_s)
np.save(os.path.join(output_dir,f'feat1_corrs_td_{tbin}_s'), feat1_corrs_td_s)

np.save(os.path.join(output_dir,f'feat2_corrs_asd_{tbin}_s'), feat2_corrs_asd_s)
np.save(os.path.join(output_dir,f'feat2_corrs_td_{tbin}_s'), feat2_corrs_td_s)

np.save(os.path.join(output_dir,f'feat3_corrs_asd_{tbin}_s'), feat3_corrs_asd_s)
np.save(os.path.join(output_dir,f'feat3_corrs_td_{tbin}_s'), feat3_corrs_td_s)

np.save(os.path.join(output_dir,f'feat0_corrs_diff_{tbin}_s'), feat0_corrs_diff_s)
np.save(os.path.join(output_dir,f'feat1_corrs_diff_{tbin}_s'), feat1_corrs_diff_s)
np.save(os.path.join(output_dir,f'feat2_corrs_diff_{tbin}_s'), feat2_corrs_diff_s)
np.save(os.path.join(output_dir,f'feat3_corrs_diff_{tbin}_s'), feat3_corrs_diff_s)


#%%
get_BootstrapPvals = lambda x: ((np.sum(x<=0)+1.)/float(x.size+1.) if x.mean()>0 else \
    (np.sum(x>=0)+1.)/float(x.size+1.))*2. # two-sided.

    
print('\n --- Spearman ---')
spearman_values = [[ feat0_corrs_asd_s.mean(), feat0_corrs_td_s.mean(), feat0_corrs_diff_s.mean() ],
                   [ feat1_corrs_asd_s.mean(), feat1_corrs_td_s.mean(), feat1_corrs_diff_s.mean() ],
                   [ feat2_corrs_asd_s.mean(), feat2_corrs_td_s.mean(), feat2_corrs_diff_s.mean() ],
                   [ feat3_corrs_asd_s.mean(), feat3_corrs_td_s.mean(), feat3_corrs_diff_s.mean() ]]

spearman_values = np.asarray(spearman_values)

print(spearman_values)


# correction for multiple comparison.
pval_comb = [[ get_BootstrapPvals(feat0_corrs_asd_s), get_BootstrapPvals(feat0_corrs_td_s), get_BootstrapPvals(feat0_corrs_diff_s) ],
             [ get_BootstrapPvals(feat1_corrs_asd_s), get_BootstrapPvals(feat1_corrs_td_s), get_BootstrapPvals(feat1_corrs_diff_s) ],
             [ get_BootstrapPvals(feat2_corrs_asd_s), get_BootstrapPvals(feat2_corrs_td_s), get_BootstrapPvals(feat2_corrs_diff_s) ],
             [ get_BootstrapPvals(feat3_corrs_asd_s), get_BootstrapPvals(feat3_corrs_td_s), get_BootstrapPvals(feat3_corrs_diff_s) ] ]

pval_comb = np.asarray(pval_comb)

fdr_TF = multipletests(pval_comb.ravel(),alpha=0.05,method='fdr_bh')[0] # fdr_bh
fdr_pvals = multipletests(pval_comb.ravel(),alpha=0.05,method='fdr_bh')[1] # fdr_bh

fdr_TF = fdr_TF.reshape(pval_comb.shape)
fdr_pvals = fdr_pvals.reshape(pval_comb.shape)

print(pval_comb)
print(fdr_TF)
print(fdr_pvals)

annot_txt = []
for rii in range(spearman_values.shape[0]):
    for cii in range(spearman_values.shape[1]):
        if fdr_pvals[rii,cii]>=0.001:
            pval_this_txt = '%1.3f'%fdr_pvals[rii,cii]
        else:
            pval_this_txt = '*'
            
        annot_txt.append('%1.3f (%s)'%(spearman_values[rii,cii],pval_this_txt))

annot_txt = [f'{tbin}'] + annot_txt

# annot_txt = np.asarray(annot_txt,dtype='str')#.reshape(spearman_values.shape)      
annot_txt = np.asarray(annot_txt,dtype='str').reshape(1,-1)      
annot_txt_df = pd.DataFrame(data=annot_txt)
annot_txt_df.to_csv(os.path.join(output_dir,f'corrs_table_{tbin}.csv'),index=False)

